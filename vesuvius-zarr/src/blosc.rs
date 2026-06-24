use derive_more::Debug;
use std::fs::File;
use std::io::{Cursor, Read};

fn zstd_decompress(input: &[u8]) -> Vec<u8> {
    let mut uncompressed = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(Cursor::new(input))
        .unwrap()
        .read_to_end(&mut uncompressed)
        .unwrap();
    uncompressed
}

#[derive(Debug, Clone)]
pub enum BloscShuffle {
    None,
    Bit,
    Byte,
}

#[derive(Debug, Clone)]
pub enum BloscCompressor {
    Blosclz,
    Lz4,
    Snappy,
    Zlib,
    Zstd,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BloscHeader {
    pub version: u8,
    pub version_lz: u8,
    pub flags: u8,
    pub typesize: usize,
    pub nbytes: usize,
    pub blocksize: usize,
    pub cbytes: usize,
    pub shuffle: BloscShuffle,
    pub compressor: BloscCompressor,
}
impl BloscHeader {
    fn from_bytes(bytes: &[u8]) -> Self {
        let flags = bytes[2];
        let shuffle = match flags & 0x7 {
            0 | 1 => BloscShuffle::None,
            2 => BloscShuffle::Byte,
            4 => BloscShuffle::Bit,
            x => panic!("Invalid shuffle value {x}"),
        };
        let compressor = match flags >> 5 {
            0 => BloscCompressor::Blosclz,
            1 => BloscCompressor::Lz4,
            2 => BloscCompressor::Snappy,
            3 => BloscCompressor::Zlib,
            4 => BloscCompressor::Zstd,
            x => panic!("Invalid compressor value {x}"),
        };

        BloscHeader {
            version: bytes[0],
            version_lz: bytes[1],
            flags,
            typesize: bytes[3] as usize,
            nbytes: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            blocksize: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            cbytes: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize,
            shuffle,
            compressor,
        }
    }

    fn num_blocks(&self) -> usize {
        if self.blocksize == 0 {
            1
        } else {
            let res = (self.nbytes + self.blocksize - 1) / self.blocksize;
            res
        }
    }
}

pub struct BloscChunk<T> {
    phantom_t: std::marker::PhantomData<T>,
}

impl BloscChunk<u8> {
    pub fn load_data_from_file(file: &File) -> Vec<u8> {
        // Read the whole chunk file in one sequential pass rather than mmapping
        // and faulting in pages on demand. On local disk both are fine, but over
        // a FUSE-backed filesystem (e.g. mountpoint-s3) mmap page faults turn
        // into many small, latency-bound reads that defeat the sequential
        // prefetcher; a single read() lets the backing store serve one large GET.
        let mut bytes = Vec::with_capacity(file.metadata().map(|m| m.len() as usize).unwrap_or(0));
        let mut reader: &File = file;
        let t_read = std::time::Instant::now();
        {
            let _g = crate::metrics::read_begin();
            reader.read_to_end(&mut bytes).unwrap();
        }
        let read_ns = t_read.elapsed().as_nanos() as u64;
        let store_bytes = bytes.len() as u64;

        let t_decode = std::time::Instant::now();
        let out = Self::decompress_to_vec(&bytes);
        crate::metrics::record_chunk_io(read_ns, t_decode.elapsed().as_nanos() as u64, store_bytes);
        out
    }

    /// In-memory variant of `load_data_from_file`: decompress a blosc-framed
    /// chunk straight from a byte slice. Used by the unified cache, which
    /// downloads chunks itself rather than going through the zarr on-disk
    /// cache.
    pub fn decompress_to_vec(bytes: &[u8]) -> Vec<u8> {
        let header = BloscHeader::from_bytes(&bytes[0..16]);
        let num_blocks = header.num_blocks();
        let mut offsets = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            offsets.push(u32::from_le_bytes([
                bytes[16 + i * 4],
                bytes[16 + i * 4 + 1],
                bytes[16 + i * 4 + 2],
                bytes[16 + i * 4 + 3],
            ]));
        }
        let mut data = Vec::with_capacity(num_blocks * header.blocksize.max(1));
        for i in 0..num_blocks {
            let block_offset = offsets[i] as usize;
            if block_offset + 4 >= bytes.len() {
                panic!("blosc block {} offset out of bounds", i);
            }
            let block_compressed_length =
                u32::from_le_bytes(bytes[block_offset..block_offset + 4].try_into().unwrap()) as usize;
            let block_compressed_data = &bytes[block_offset + 4..block_offset + block_compressed_length + 4];
            let block = match header.compressor {
                BloscCompressor::Lz4 => lz4_compression::decompress::decompress(block_compressed_data)
                    .unwrap_or_else(|_| vec![0; header.blocksize]),
                BloscCompressor::Zstd => zstd_decompress(block_compressed_data),
                _ => panic!("Unsupported blosc compressor: {:?}", header.compressor),
            };
            data.extend(block);
        }
        data
    }
}
