use std::simd::prelude::*;

pub fn simd_bf16_to_f32(bf16: u16x64) -> f32x64 {
    let is_nan =
        mask32x64::from((bf16 & u16x64::splat(0x7FFFu16)).simd_gt(u16x64::splat(0x7F80u16)));
    let bf16_u32 = bf16.cast::<u32>();
    let is_nan_result = bf16_u32 | u32x64::splat(0x0040u32);

    f32x64::from_bits(is_nan.select(is_nan_result, bf16_u32) << 16)
}

pub fn simd_f32_to_bf16(f32: f32x64) -> u16x64 {
    let x = f32.to_bits();

    let is_nan =
        mask16x64::from((x & u32x64::splat(0x7FFF_FFFFu32)).simd_gt(u32x64::splat(0x7F80_0000u32)));
    let nan_result = ((x >> 16) | u32x64::splat(0x0040u32)).cast::<u16>();

    let round_bit = u32x64::splat(1 << 15);
    let zero = u32x64::splat(0);
    let round_mask =
        (x & round_bit).simd_ne(zero) & (x & (round_bit + round_bit << 1 - 1)).simd_ne(zero);
    let rounded_result = round_mask.select(x >> 16, x >> 16 + 1).cast::<u16>();
    is_nan.select(nan_result, rounded_result)
}
