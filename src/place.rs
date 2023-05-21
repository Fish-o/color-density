use crate::{HashMapFnv, HashSetFnv, Offset, Pos, Solution};

pub fn place_color(
  available_spaces: &mut HashSetFnv<Pos>,
  solution: &mut Solution,
  pos: Pos,
  color: u8,
  distance_offsets: &HashMapFnv<u8, Vec<Offset>>,
) {
  solution.add(pos.clone(), color);
  available_spaces.remove(&pos);
  let offsets = distance_offsets.get(&color).expect("Color not distanced");

  for offset in offsets {
    let new_pos = pos.offset(offset);
    if let Some(pos) = new_pos {
      if available_spaces.contains(&pos) {
        available_spaces.remove(&pos);
      }
    }
  }
}
