#![feature(unsafe_cell_from_mut)]
#![feature(write_all_vectored)]

#[macro_use]
extern crate log;

use std::cell::UnsafeCell;
use std::fs::OpenOptions;
use std::io;
use std::io::{IoSlice, Read, Write};
use std::path::Path;
use std::ptr::null_mut;
use std::time::Instant;
use crossbeam_channel::Sender;

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

fn create_device(
    buff_size: usize,
    device_batch: usize,
    pool: &Sender<Device>
) {
    let mut count = 0;

    unsafe {
        cudaGetDeviceCount(&mut count);
    }

    for _ in 0..device_batch {
        for i in 0..count {
            let device = Device::new(i, buff_size).unwrap();
            pool.send(device).unwrap();
        }
    }
}


#[inline]
fn trim_to_fr32(buff: &mut [u8; 32]) {
    // strip last two bits, to ensure result is in Fr.
    buff[31] &= 0b0011_1111;
}

struct Wrap<T>(T);

unsafe impl<T> Send for Wrap<T> {}

unsafe impl<T> Sync for Wrap<T> {}

/// Build TreeD with GPU
///
/// # Parameters
///
/// - `in_path`: unsealed file
/// - `out_path`: TreeD file
/// - `buffer`: require buffer size >= unsealed-file-size * 2 - 32
/// - `cuda_memory_limit`: maximum GPU memory used, unit is byte, power of tow
///
/// # Returns
///
/// TreeD root hash
pub fn build_treed(
    in_path: &Path,
    out_path: &Path,
    buffer: &mut [u8],
    cuda_memory_limit: usize
) -> io::Result<[u8; 32]> {
    if !cuda_memory_limit.is_power_of_two() {
        return Err(io::Error::new(io::ErrorKind::Other, "cuda memory limit must power of 2"));
    }

    let chunk_size = cuda_memory_limit / 2 / 2;
    let buffer = &Wrap(&*UnsafeCell::from_mut(buffer));

    let mut in_file = std::fs::File::open(in_path)?;
    let file_size = in_file.metadata()?.len() as usize;

    if !file_size.is_power_of_two() || file_size < chunk_size {
        return Err(io::Error::new(io::ErrorKind::Other, "file size not match"));
    }

    let mut out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(out_path)?;

    let (device_pool_tx, device_pool_rx) = crossbeam_channel::unbounded();
    let (base_data_write, base_data_read) = crossbeam_channel::unbounded();
    let (tree_data_write, tree_data_read) = crossbeam_channel::unbounded();

    let chunks = file_size / chunk_size;

    std::thread::scope(|s| {
        s.spawn(|| {
            create_device(chunk_size * 2 - 32, 2, &device_pool_tx);
        });

        let base_data_writer = s.spawn(move || {
            let t = Instant::now();

            while let Ok(slice) = base_data_read.recv() {
                out_file.write_all(slice).unwrap();
            };

            info!("finish to write base data, use: {:?}", t.elapsed());
            out_file
        });

        let mut offset = 0;
        let mut chunk_index = 0;

        let t = Instant::now();

        while chunk_index < chunks {
            let chunk = &mut unsafe { &mut *buffer.0.get() }[offset..offset + chunk_size];
            in_file.read_exact(chunk)?;

            let i = Wrap(chunk_index);

            s.spawn(|| {
                let i = i;
                let i = i.0;
                let offset = i * (chunk_size * 2 - 32);

                let buffer = buffer;
                let buff = unsafe { &mut *buffer.0.get() };
                let tree_data = &mut buff[offset..offset + chunk_size * 2 - 32];

                let device = device_pool_rx.recv().unwrap();
                let t = Instant::now();
                unsafe { build_tree(device.index, tree_data.as_mut_ptr(), device.buff, chunk_size / 32) };
                info!("build_tree use: {:?}", t.elapsed());
                device_pool_tx.send(device).unwrap();

                tree_data_write.send((i, &tree_data[chunk_size..])).unwrap();
            });

            base_data_write.send(&*chunk).unwrap();
            offset += chunk_size * 2 - 32;
            chunk_index += 1;
        }

        info!("read base_data use: {:?}", t.elapsed());

        drop(base_data_write);
        let mut out_file = base_data_writer.join().unwrap();

        let t = Instant::now();

        let mut expect_index = 0usize;
        let mut chunk_size = chunk_size / 2;
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

        loop {
            for node in nodes.as_slice() {
                out_file.write_all(node)?;
            }

            if nodes.len() == 1 {
                info!("build total sub_tree use: {:?}", t.elapsed());
                return Ok(nodes[0]);
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
    })
}