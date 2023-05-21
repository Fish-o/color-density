#[macro_use]
extern crate lazy_static;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::ops::{RangeBounds, RangeInclusive};
use std::path::Path;
use std::process::exit;
use std::{
  collections::{HashMap, HashSet},
  sync::mpsc::channel,
  time::Duration,
};
mod hasher;
mod place;
use hasher::FnvHasher;
use place::place_color;
use std::io::{Read, Write};
use threadpool::ThreadPool;

// Variables to tinker with:
const SIZE: usize = 32;
const SECS: u64 = 1;
const THREADS: usize = 16; // I used 16 threads for every test
const COLORS: RangeInclusive<u8> = 2..=16;

// SIZE 4 PROCEDURE 62/64 (97%)
//  1 generated 1 sec 2297484/s
//  2-16 generated 1 sec 301754/s

// SIZE 8 PROCEDURE 408/512 (80%)
//  1 manually
//  2 generated 10 min 489001/s
//  3-4 generated 5 min 167055/s
//  5-9 generated 3 min 95520/s
//  10-16 generated 5 min 102858/s

// SIZE 16 PROCEDURE 2935/4096 (72%)
//  1 manually
//  2 generated 15 min 34347/s
//  3 generated 10 min 21969/s
//  4-5 generated 5 min 25239/s
//  6-10 generated 5 min 13637/s
//  11-16 generated 5 min 12132/s

// 4 8 16 32
// Max 16 colors

const MAX_SIZE: u64 = (SIZE * SIZE * SIZE) as u64;
lazy_static! {
  static ref COLOR_BEFORE: u8 = COLORS.clone().next().or(Some(1)).unwrap() - 1;
  static ref LOADED_DATA: (Vec<Pos>, HashMapFnv<u8, HashSetFnv<Pos>>) = load(SIZE, *COLOR_BEFORE);
  static ref FREE_POSITIONS: Vec<Pos> = LOADED_DATA.0.clone();
  static ref ALL_AVAILABLE: HashSetFnv<Pos> = FREE_POSITIONS.clone().into_iter().collect();
  static ref VALS: HashMapFnv<u8, HashSetFnv<Pos>> = LOADED_DATA.1.clone();
  // How to store 3 u8s in a u64
  // Size is 8, which can be stored using 3 bits
  // 3 * 3 = 9
  // 0b00000000000000000000000000000000
  // 0b10000000000000xxx000yyy000zzz000, first bit is for underflow protection
  // 0b01111111111111000111000111000111, this masks the values that have to be zero for it to be a valid position

  static ref GOOD_MASK: u32 = if SIZE == 4 {
    0b01111111001111111001111111001111
  } else if SIZE == 8 {
    0b01111110001111110001111110001111
  } else if SIZE == 16 {
    0b01111100001111100001111100001111
  } else if SIZE == 32 {
    0b01111000001111000001111000001111
  } else {
    panic!("Invalid SIZE, only 4, 8, 16 and 32 allowed")
  } as u32;

}

fn load(size: usize, colors: u8) -> (Vec<Pos>, HashMapFnv<u8, HashSetFnv<Pos>>) {
  let mut free_positions = Vec::new();

  let mut vals: HashMapFnv<u8, HashSetFnv<Pos>> = HashMapFnv::default();

  let path = Path::new("solutions")
    .join(format!("SIZE_{}", size))
    .join(format!("COLORS_{}.txt", colors));
  // If path doesnt exist then return empty
  if colors == 0 {
    let free_positions = vec![0; SIZE * SIZE * SIZE]
      .into_iter()
      .enumerate()
      .map(|(i, _)| Pos::from(i % SIZE, (i / SIZE) % SIZE, i / (SIZE * SIZE)))
      .collect::<Vec<Pos>>();
    return (free_positions, vals);
  } else if !path.exists() {
    println!("Path: {:?}", path);
    println!(
      "ERROR: Missing solution for size {} and color {}",
      size, colors
    );
    exit(1);
  }

  let mut file = File::open(path).unwrap();
  let mut layers = String::new();
  file.read_to_string(&mut layers).unwrap();

  for (z, layer) in layers
    .split("|")
    .filter(|s| !s.starts_with("--"))
    .filter(|r| !r.is_empty())
    .enumerate()
  {
    // println!("Loading layer {}", z);
    // println!("Layer: {}", layer);
    for (y, row) in layer.split('\n').filter(|r| !r.is_empty()).enumerate() {
      // println!("Row: {}", row);
      for (x, c) in row.chars().enumerate() {
        // println!("Char: '{}' {} {} {}", c, x, y, z);
        if c == ' ' {
          free_positions.push(Pos::from(x, y, z));
        } else {
          vals
            .entry(match c.to_ascii_lowercase() {
              '1' => 1,
              '2' => 2,
              '3' => 3,
              '4' => 4,
              '5' => 5,
              '6' => 6,
              '7' => 7,
              '8' => 8,
              '9' => 9,
              'a' => 10,
              'b' => 11,
              'c' => 12,
              'd' => 13,
              'e' => 14,
              'f' => 15,
              'g' => 16,
              _ => panic!("Invalid char"),
            })
            .or_insert(HashSetFnv::default())
            .insert(Pos::from(x, y, z));
        }
      }
    }
  }
  // println!("Loaded {} positions", positions.len());
  // println!("Loaded {} vals", vals.len());
  (free_positions, vals)
}

#[derive(Clone, Copy, Debug)]
pub struct Pos(pub u32);
impl Pos {
  pub fn from(x: usize, y: usize, z: usize) -> Self {
    let val: u32 = (x << 22 | y << 13 | z << 4 | 1 << 31) as u32;
    // println!("{:b}", val);
    if val & *GOOD_MASK != 0 {
      panic!("Invalid position");
    }
    Self(val)
  }

  pub fn val(&self) -> (usize, usize, usize) {
    let val = self.0;
    let x = (val >> 22) & 0b011111;
    let y = (val >> 13) & 0b011111;
    let z = (val >> 4) & 0b011111;
    (x as usize, y as usize, z as usize)
  }
  pub fn offset(&self, offset: &Offset) -> Option<Self> {
    let val = self.0;
    let n = val + offset.0 - offset.1;
    if n & *GOOD_MASK != 0 {
      return None;
    }
    Some(Self(n))
  }
}

impl PartialEq for Pos {
  fn eq(&self, other: &Self) -> bool {
    self.0 == other.0
  }
}
impl Eq for Pos {}
impl Hash for Pos {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state);
  }
}
#[derive(Clone, Copy, Debug)]
pub struct Offset(u32, u32);
impl Offset {
  pub fn from(x: i64, y: i64, z: i64) -> Option<Self> {
    let mut pos_val: u32 = 0;
    let mut neg_val: u32 = 0;
    if x.is_positive() {
      pos_val |= (x.abs() as u32) << 22;
    } else {
      neg_val |= (x.abs() as u32) << 22;
    }
    if y.is_positive() {
      pos_val |= (y.abs() as u32) << 13;
    } else {
      neg_val |= (y.abs() as u32) << 13;
    }
    if z.is_positive() {
      pos_val |= (z.abs() as u32) << 4;
    } else {
      neg_val |= (z.abs() as u32) << 4;
    }

    // println!("Pos {:b}", pos_val);
    // println!("Neg {:b}", neg_val);
    if pos_val & *GOOD_MASK != 0 {
      // println!("x: {}, y: {}, z: {}", x, y, z);
      // println!("{:b}", pos_val);
      // panic!("Invalid pos");
      None
    } else if neg_val & *GOOD_MASK != 0 {
      // println!("{:b}", neg_val);
      // panic!("Invalid neg");
      None
    } else {
      Some(Self(pos_val, neg_val))
    }
  }
}
impl Hash for Offset {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state);
    self.1.hash(state);
  }
}
type HashMapFnv<K, V> = HashMap<K, V, BuildHasherDefault<FnvHasher>>;
type HashSetFnv<K> = HashSet<K, BuildHasherDefault<FnvHasher>>;

#[derive(Clone, Debug)]
pub struct Solution {
  placed: HashMapFnv<u8, HashSetFnv<Pos>>,
}

impl Solution {
  pub fn new() -> Self {
    let vals: HashMapFnv<u8, HashSetFnv<Pos>> = VALS.clone();
    // let mut ones = HashSetFnv::<Pos>::default();

    // for z in 0..4 {
    //   let mut x_start = 0;
    //   if z % 2 == 0 {
    //     x_start = 1;
    //   }
    //   for y in 0..4 {
    //     let mut x_start = x_start;
    //     if y % 2 == 0 {
    //       x_start += 1;
    //       x_start = x_start % 2;
    //     }
    //     ones.insert(Pos::from(x_start, y, z));
    //     ones.insert(Pos::from(x_start + 2, y, z));
    //   }
    // }
    // vals.insert(1, ones);
    Self { placed: vals }
  }
  pub fn from(vals: HashMapFnv<u8, HashSetFnv<Pos>>) -> Self {
    Self { placed: vals }
  }
  pub fn add(&mut self, pos: Pos, value: u8) {
    self
      .placed
      .entry(value)
      .or_insert(HashSetFnv::default())
      .insert(pos);
  }
  pub fn print(&self) -> String {
    let mut res: String = String::new();
    for z in 0..SIZE {
      res += &format!("|--Layer {}|\n", z);
      for y in 0..SIZE {
        for x in 0..SIZE {
          let mut found = false;
          for (value, placed) in self.placed.iter() {
            if placed.contains(&Pos::from(x, y, z)) {
              let char = match value {
                0 => '?',
                1 => '1',
                2 => '2',
                3 => '3',
                4 => '4',
                5 => '5',
                6 => '6',
                7 => '7',
                8 => '8',
                9 => '9',
                10 => 'A',
                11 => 'B',
                12 => 'C',
                13 => 'D',
                14 => 'E',
                15 => 'F',
                16 => 'G',
                _ => ' ',
              };
              res += &format!("{}", char);
              found = true;
              break;
            }
          }
          if !found {
            res += &format!(" ");
          }
        }
        res += &format!("\n");
      }
    }
    res
  }
  pub fn score(&self) -> u64 {
    let mut density: u64 = 0;
    for (_, placed) in self.placed.iter() {
      density += placed.len() as u64;
    }
    density
  }
}

fn create_distance_offsets(up_to: u8) -> HashMapFnv<u8, Vec<Offset>> {
  let mut offsets: HashMapFnv<u8, Vec<Offset>> = HashMapFnv::default();
  for color in 1..=up_to {
    let mut offsets_vec: Vec<(i16, i16, i16)> = vec![];
    for x in 0..=color {
      for y in 0..=color {
        for z in 0..=color {
          if x + y + z == color {
            offsets_vec.push((x as i16, y as i16, z as i16));
          }
        }
      }
    }

    offsets_vec.extend(offsets_vec.clone().iter().map(|(x, y, z)| (-*x, *y, *z)));
    offsets_vec.extend(offsets_vec.clone().iter().map(|(x, y, z)| (*x, -*y, *z)));
    offsets_vec.extend(offsets_vec.clone().iter().map(|(x, y, z)| (*x, *y, -*z)));

    let mut final_offset_vec = offsets_vec
      .iter()
      .map(|(x, y, z)| Offset::from(*x as i64, *y as i64, *z as i64))
      .filter_map(|x| x)
      .collect::<Vec<_>>();

    if let Some(offset) = offsets.get(&(&color - 1)) {
      final_offset_vec.extend(offset.clone());
    }
    offsets.insert(color, final_offset_vec);
  }
  offsets
}
fn all_available_spaces(solution: &Solution) -> HashSetFnv<Pos> {
  let mut available_spaces = ALL_AVAILABLE.clone();
  for (_, placed) in solution.placed.iter() {
    for pos in placed.iter() {
      available_spaces.remove(pos);
    }
  }

  available_spaces
}

fn fill(
  colors: impl RangeBounds<u8> + Clone + IntoIterator<Item = u8>,
  distance_offsets: &HashMapFnv<u8, Vec<Offset>>,
) -> Solution {
  let mut solution = Solution::new();
  for color in colors {
    let mut available_spaces = all_available_spaces(&solution);
    loop {
      if available_spaces.is_empty() {
        break;
      }
      // Pick random space
      let length = available_spaces.len();
      let index = rand::random::<usize>() % length;
      let pos = available_spaces.iter().nth(index).unwrap().clone();

      place_color(
        &mut available_spaces,
        &mut solution,
        pos,
        color,
        &distance_offsets,
      );
    }
  }
  solution
}

fn display_best(duration: &Duration, score: &u64, thread: usize) {
  println!(
    "T{:0>2} {:.4} - New: {}/{}",
    thread,
    duration.as_secs_f32(),
    score,
    MAX_SIZE
  );
}

fn solve(
  distance_offsets: &HashMapFnv<u8, Vec<Offset>>,
  max_duration: &Duration,
  colors: impl RangeBounds<u8> + Clone + IntoIterator<Item = u8>,
  thread: usize,
) -> (i32, Solution) {
  let time = std::time::Instant::now();
  let mut best_sol = fill(colors.clone(), &distance_offsets);
  let mut best_score = best_sol.score();
  let mut iterations = 0;
  loop {
    iterations += 1;
    if &time.elapsed() > max_duration {
      break;
    }
    let sol = fill(colors.clone(), &distance_offsets);
    let score = sol.score();
    if score > best_score {
      best_score = score;
      best_sol = sol;
      display_best(&time.elapsed(), &best_score, thread);
      if best_score == MAX_SIZE {
        break;
      }
    }
  }
  (iterations, best_sol)
}

fn multi_solve(
  distance_offsets: &HashMapFnv<u8, Vec<Offset>>,
  colors: impl RangeBounds<u8> + Clone + IntoIterator<Item = u8> + std::marker::Send + 'static,
) {
  let n_workers = THREADS;
  let pool = ThreadPool::new(n_workers);
  let (tx, rx) = channel::<(i32, Solution)>();
  let start = std::time::Instant::now();
  for thread in 0..n_workers {
    let tx = tx.clone();
    let distance_offsets = distance_offsets.clone();
    let colors = colors.clone();
    pool.execute(move || {
      let sol = solve(
        &distance_offsets,
        &Duration::from_secs(SECS),
        colors,
        thread,
      );
      tx.send(sol).unwrap();
    });
  }
  let responses = rx.iter().take(n_workers).collect::<Vec<_>>();
  let duration = start.elapsed();
  let iterations = responses.iter().map(|x| x.0).sum::<i32>();
  let responses = responses.iter().map(|x| x.1.clone()).collect::<Vec<_>>();
  println!(
    "Iterations: {:.0}/s",
    iterations as f64 / duration.as_secs_f64()
  );
  let best = responses
    .iter()
    .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap())
    .unwrap();
  let solved_rate = responses
    .iter()
    .filter(|x| x.score() == best.score())
    .count();

  let color_end = colors.clone().into_iter().last().or(Some(0)).unwrap();
  println!(
    "C{:0>2} Solution: {}/{} ({:.3}%)",
    color_end,
    best.score(),
    MAX_SIZE,
    best.score() as f32 / MAX_SIZE as f32 * 100.0
  );
  println!("Same score reached by: {}/{}", solved_rate, n_workers);
  store(&best, color_end);
}

fn store(solution: &Solution, colors: u8) {
  let mut path = Path::new("./solutions")
    .join(format!("SIZE_{}", SIZE))
    .join(format!("COLORS_{}.txt", colors));
  // Create dirs if they don't exists
  if let Some(parent) = path.parent() {
    if !parent.exists() {
      std::fs::create_dir_all(parent).unwrap();
    }
  }
  if path.exists() {
    let (_, vals) = load(SIZE, colors);
    let stored_sol = Solution::from(vals);
    if stored_sol.score() as u64 > solution.score() {
      println!(
        "Better solution already exists ({}/{}, {:.2}%), storing as .worse",
        stored_sol.score(),
        MAX_SIZE,
        stored_sol.score() as f32 / MAX_SIZE as f32 * 100.0
      );
      path.set_extension("worse");
    }
  }
  let mut file = File::create(&path).unwrap();
  file.write_all(solution.print().as_bytes()).unwrap();
  println!("Stored at {:?}", path);
}
fn main() {
  println!("Creating offsets...");
  let distance_offsets = create_distance_offsets(16);
  println!("Done!");
  let colors = COLORS;
  multi_solve(&distance_offsets, colors);

  // To load in everything properly make sure to set SIZE const to 16!!!
  // mc_place_commands(4, 0);
  // mc_place_commands(4, 1);
  // mc_place_commands(8, 0);
  // mc_place_commands(8, 1);
  // mc_place_commands(16, 0);
  // mc_place_commands(16, 1);
}

fn _mc_place_commands(size: usize, gap: usize) {
  let loaded = load(size, 16);
  let free = loaded.0;
  let placed = loaded.1;
  let mut functions = String::new();
  let math_gap = gap + 1;

  for (color, positions) in placed.iter() {
    for pos in positions {
      let (x, y, z) = pos.val();
      let x = x * math_gap;
      let y = y * math_gap;
      let z = z * math_gap;
      let x = x + 3;
      let block_color = match color {
        1 => "white",
        2 => "light_gray",
        3 => "gray",
        4 => "black",
        5 => "brown",
        6 => "red",
        7 => "orange",
        8 => "yellow",
        9 => "lime",
        10 => "green",
        11 => "cyan",
        12 => "light_blue",
        13 => "blue",
        14 => "purple",
        15 => "magenta",
        16 => "pink",
        _ => unreachable!(),
      };
      let block = format!("minecraft:{}_concrete", block_color);
      functions.push_str(&format!("setblock ~{} ~{} ~{} {}\n", x, y, z, block));
    }
  }
  for pos in free {
    let (x, y, z) = pos.val();
    let x = x * math_gap;
    let y = y * math_gap;
    let z = z * math_gap;
    let x = x + 3;
    functions.push_str(&format!("setblock ~{} ~{} ~{} minecraft:cobweb\n", x, y, z));
  }
  let path = Path::new("./datapack/data/blockplacer/functions")
    .join(format!("size{}gap{}.mcfunction", size, gap));
  let mut file = File::create(&path).unwrap();
  file.write_all(functions.as_bytes()).unwrap();
  println!("Stored at {:?}", path);
}
