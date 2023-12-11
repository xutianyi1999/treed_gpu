use std::fs::OpenOptions;
use std::io;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Condvar, Mutex};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};

#[allow(unused)]
const LEAF: u8 = 0x00;
const INTERIOR: u8 = 0x01;

extern "C" {
    fn build_tree(device: i32, tree_data: *mut u8, nodes: usize);

    fn cudaGetDeviceCount(count: &mut i32);
}

struct Device {
    index: i32,
    limit: usize,
    pair: (Mutex<usize>, Condvar),
}

impl Device {
    fn new(index: i32, limit: usize) -> Device {
        Device {
            index,
            limit,
            pair: (Mutex::new(0), Condvar::new()),
        }
    }

    fn sync(&self) {
        let limit = self.limit;
        let (lock, cvar) = &self.pair;

        let mut limit_guard = lock.lock().unwrap();
        limit_guard = cvar.wait_while(limit_guard, |v| *v >= limit).unwrap();
        *limit_guard += 1;
    }

    fn release(&self) {
        let (lock, cvar) = &self.pair;

        let mut limit_guard = lock.lock().unwrap();
        *limit_guard -= 1;

        cvar.notify_one();
    }
}

pub fn build_treed(in_path: &Path, out_path: &Path) -> io::Result<()> {
    let mut count = 0;

    unsafe {
        cudaGetDeviceCount(&mut count);
    };

    if count == 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "must need cuda gpu"));
    }

    let device_list = (0..count).map(|index| Device::new(index, 4)).collect::<Vec<_>>();

    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(out_path)?;
    let out_file = Mutex::new(out_file);

    let file_size = std::fs::metadata(in_path)?.len();
    const CHUNK_SIZE: u64 = 512 * 1024 * 1024;

    // todo remove the requirement that the file must be >= CHUNK_SIZE
    if !file_size.is_power_of_two() || file_size < CHUNK_SIZE {
        return Err(io::Error::new(io::ErrorKind::Other, "file size not match"));
    }

    let chunks = file_size / CHUNK_SIZE;

    let res = (0..chunks).into_par_iter()
        .map(|i| {
            let device = &device_list[i as usize % device_list.len()];
            let mut tree_data = vec![0u8; (CHUNK_SIZE * 2 - 32) as usize];

            let mut in_file = std::fs::File::open(in_path)?;
            in_file.seek(SeekFrom::Start(i * CHUNK_SIZE))?;
            in_file.read_exact(&mut tree_data[..CHUNK_SIZE as usize])?;

            device.sync();
            unsafe { build_tree(device.index, tree_data.as_mut_ptr(), CHUNK_SIZE as usize / 32) };
            device.release();

            let mut out_file = out_file.lock().unwrap();

            let mut chunk_size = CHUNK_SIZE;
            let mut offset = 0;
            let mut next_buff = tree_data.as_slice();

            out_file.seek(SeekFrom::Start(i * chunk_size))?;

            while chunk_size > 32 {
                out_file.write_all(&next_buff[..chunk_size as usize])?;
                offset += chunks * chunk_size;
                next_buff = &next_buff[chunk_size as usize..];
                chunk_size /= 2;
                out_file.seek(SeekFrom::Start(offset + i * chunk_size))?;
            }

            let node: &[u8; 32] = (&next_buff[..chunk_size as usize]).try_into().unwrap();
            out_file.write_all(node)?;
            Ok(*node)
        })
        .collect::<io::Result<Vec<[u8; 32]>>>();

    let mut node_list = res?;
    let mut out_file = out_file.lock().unwrap();

    while node_list.len() > 0 {
        for node in node_list.as_slice() {
            out_file.write_all(node)?;
        }

        node_list = node_list.chunks_exact(2).map(|nodes| {
            let left = &nodes[0];
            let right = &nodes[1];

            let mut hasher = Sha256::new();
            hasher.update(&[INTERIOR]);
            hasher.update(left);
            hasher.update(right);
            let out = hasher.finalize();
            let out: &[u8; 32] = out.as_slice().try_into().unwrap();
            *out
        }).collect();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn get_gpus() {
        let mut count = 0;
        unsafe { cudaGetDeviceCount(&mut count) };
        println!("{}", count);
    }

    #[test]
    fn it_works() {
        let t = Instant::now();
        let gpu_tree_data = vec![];
        println!("gpu: {:?}", t.elapsed());

        let t = Instant::now();

        let mut tree_data = vec![0u8; 1024 * 1024 * 32];

        for slice in tree_data.chunks_mut(32) {
            let mut hasher = Sha256::new();
            hasher.update(&[LEAF]);
            hasher.update(&*slice);
            let res = hasher.finalize();
            slice.copy_from_slice(&res);
        }

        let mut data = tree_data.clone();
        let mut label: Vec<u8> = Vec::with_capacity(tree_data.len() / 2);

        loop {
            for slice in data.chunks(64) {
                let mut hasher = Sha256::new();
                hasher.update(&[INTERIOR]);
                hasher.update(slice);
                let res = hasher.finalize();
                label.extend_from_slice(&res);
            }

            tree_data.extend_from_slice(&label);

            if label.len() == 32 {
                break;
            }
            data = label.clone();
            label.clear();
        }

        println!("cpu: {:?}", t.elapsed());

        println!("{}, {}", tree_data.len() / 32, gpu_tree_data.len() / 32);
        // assert_eq!(&label[262144 * 32],&gpu_tree_data[262144 * 32])
        // assert!(&tree_data[..(1024 * 1024) * 32] == &gpu_tree_data[..(1024 * 1024) * 32])
        assert!(&tree_data == &gpu_tree_data);
    }
}
