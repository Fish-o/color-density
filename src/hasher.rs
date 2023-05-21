use std::hash::Hasher;

const INITIAL_STATE: u64 = 0xcbf29ce484222325;
const PRIME: u64 = 0x100000001b3;

pub struct FnvHasher(u64);

impl Default for FnvHasher {
  #[inline]
  fn default() -> FnvHasher {
    FnvHasher(INITIAL_STATE)
  }
}

// impl FnvHasher {
//   /// Create an FNV hasher starting with a state corresponding
//   /// to the hash `key`.
//   #[inline]
//   pub fn with_key(key: u64) -> FnvHasher {
//     FnvHasher(key)
//   }
// }

impl Hasher for FnvHasher {
  #[inline]
  fn finish(&self) -> u64 {
    self.0
  }

  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    let FnvHasher(mut hash) = *self;

    for byte in bytes.iter() {
      hash = hash ^ (*byte as u64);
      hash = hash.wrapping_mul(PRIME);
    }

    *self = FnvHasher(hash);
  }
}
