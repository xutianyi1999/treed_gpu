#![feature(unsafe_cell_from_mut)]
#![feature(write_all_vectored)]

use std::cell::UnsafeCell;
use std::fs::OpenOptions;
use std::io;
use std::io::{IoSlice, Read, Write};
use std::path::Path;
use std::ptr::null_mut;

use sha2::{Digest, Sha256};

extern "C" {
    fn build_tree(device: i32, tree_data: *mut u8, cuda_tree_data: *mut u8, nodes: usize);

    fn cudaGetDeviceCount(count: &mut i32);

    fn cudaSetDevice(device: i32);

    fn cudaMalloc(ptr: &mut *mut u8, size: usize);

    fn cudaFree(ptr: *mut u8);
}

struct Device {
    index: i32,
    buff: &'static mut u8,
}

impl Device {
    fn new(index: i32, buff_size: usize) -> io::Result<Device> {
        unsafe {
            cudaSetDevice(index);

            let mut ptr = null_mut();
            cudaMalloc(&mut ptr, buff_size);

            if ptr.is_null() {
                return Err(io::Error::new(io::ErrorKind::Other, "can not alloc cuda mem"));
            }

            let d = Device {
                index,
                buff: &mut *ptr,
            };
            Ok(d)
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { cudaFree(self.buff) }
    }
}

fn create_device(buff_size: usize, device_batch: usize) -> io::Result<Vec<Device>> {
    let mut count = 0;
    unsafe {
        cudaGetDeviceCount(&mut count);
    }

    let mut device_list = Vec::with_capacity(count as usize * device_batch);

    for _ in 0..device_batch {
        for i in 0..count {
            let device = Device::new(i, buff_size)?;
            device_list.push(device);
        }
    }

    Ok(device_list)
}


#[inline]
fn trim_to_fr32(buff: &mut [u8; 32]) {
    // strip last two bits, to ensure result is in Fr.
    buff[31] &= 0b0011_1111;
}

const CHUNK_SIZE: usize = 2 * 1024 * 1024 * 1024;

struct Wrap<T>(T);

unsafe impl<T> Send for Wrap<T> {}

unsafe impl<T> Sync for Wrap<T> {}

pub fn build_treed(
    in_path: &Path,
    out_path: &Path,
    buffer: &mut [u8],
) -> io::Result<()> {
    let buffer = &Wrap(&*UnsafeCell::from_mut(buffer));

    let mut in_file = std::fs::File::open(in_path)?;
    let file_size = in_file.metadata()?.len() as usize;

    if !file_size.is_power_of_two() || file_size < CHUNK_SIZE {
        return Err(io::Error::new(io::ErrorKind::Other, "file size not match"));
    }

    let mut out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(out_path)?;

    let (device_pool_tx, device_pool_rx) = crossbeam_channel::unbounded();
    let (base_data_write, base_data_read) = crossbeam_channel::unbounded();
    let (tree_data_write, tree_data_read) = crossbeam_channel::unbounded();

    let chunks = file_size / CHUNK_SIZE;

    std::thread::scope(|s| {
        s.spawn(|| {
            let devices = create_device(CHUNK_SIZE, 2).unwrap();

            for d in devices {
                device_pool_tx.send(d).unwrap();
            }
        });

        let base_data_writer = s.spawn(move || {
            while let Ok(slice) = base_data_read.recv() {
                out_file.write_all(slice).unwrap();
            };
            out_file
        });

        let mut offset = 0;
        let mut chunk_index = 0;

        while chunk_index < chunks {
            let chunk = &mut unsafe { &mut *buffer.0.get() }[offset..offset + CHUNK_SIZE];
            in_file.read_exact(chunk)?;

            let i = Wrap(chunk_index);

            s.spawn(|| {
                let i = i;
                let i = i.0;
                let offset = i * (CHUNK_SIZE * 2 - 32);

                let buffer = buffer;
                let buff = unsafe { &mut *buffer.0.get() };
                let tree_data = &mut buff[offset..offset + CHUNK_SIZE * 2 - 32];

                let device = device_pool_rx.recv().unwrap();
                unsafe { build_tree(device.index, tree_data.as_mut_ptr(), device.buff, CHUNK_SIZE / 32) };
                device_pool_tx.send(device).unwrap();

                tree_data_write.send((i, &tree_data[CHUNK_SIZE..])).unwrap();
            });

            base_data_write.send(&*chunk).unwrap();
            offset += CHUNK_SIZE * 2 - 32;
            chunk_index += 1;
        }

        drop(base_data_write);
        let mut out_file = base_data_writer.join().unwrap();

        let mut expect_index = 0usize;
        let mut chunk_size = CHUNK_SIZE / 2;
        let mut sub_tree_data = Vec::with_capacity(chunks);

        while expect_index < chunks {
            let (index, buff) = tree_data_read.recv().unwrap();

            sub_tree_data.push((index, buff));
            sub_tree_data.sort_unstable_by_key(|(i, _)| *i);

            while sub_tree_data.get(expect_index).map(|(i, _)| *i) == Some(expect_index) {
                let sub_tree_buff = &mut sub_tree_data[expect_index].1;

                out_file.write_all(&sub_tree_buff[..chunk_size])?;
                *sub_tree_buff = &sub_tree_buff[chunk_size..];
                expect_index += 1;
            }
        }

        chunk_size /= 2;

        let mut layer = Vec::with_capacity(sub_tree_data.len());

        while chunk_size > 32 {
            for (_, slice) in sub_tree_data.as_mut_slice() {
                layer.push(IoSlice::new(&slice[..chunk_size]));
                *slice = &slice[chunk_size..];
            }

            out_file.write_all_vectored(&mut layer)?;
            layer.clear();
            chunk_size /= 2;
        }

        let mut nodes: Vec<[u8; 32]> = sub_tree_data.into_iter().map(|(_, buff)| {
            let slice: &[u8; 32] = buff.try_into().unwrap();
            *slice
        }).collect();

        while !nodes.is_empty() {
            for node in nodes.as_slice() {
                out_file.write_all(node)?;
            }

            nodes = nodes.chunks_exact(2).map(|nodes| {
                let left = &nodes[0];
                let right = &nodes[1];

                let mut hasher = Sha256::new();
                hasher.update(left);
                hasher.update(right);
                let mut out = hasher.finalize();
                let out: &mut [u8; 32] = out.as_mut_slice().try_into().unwrap();
                trim_to_fr32(out);
                *out
            }).collect();
        }
        Ok(())
    })
}