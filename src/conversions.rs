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

    let round_bit = u32x64::splat(0x0000_8000u32);
    let round_bit_3 = u32x64::splat(0x0001_7fffu32);
    let zero = u32x64::splat(0);
    let round_mask = (x & round_bit).simd_ne(zero) & (x & round_bit_3).simd_ne(zero);
    let rounded_result = round_mask
        .select((x >> 16) + u32x64::splat(1), x >> 16)
        .cast::<u16>();
    is_nan.select(nan_result, rounded_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn check_f32_to_bf16(num: f32) {
        let simd_f32 = f32x64::splat(num);
        let simd_bf16 = simd_f32_to_bf16(simd_f32);
        simd_bf16.to_array().iter().for_each(|&x| {
            let bf16_num = bf16::from_f32(num);
            let bf16_x = bf16::from_bits(x);
            if bf16_num.is_nan() {
                assert!(bf16_x.is_nan());
            } else {
                assert_eq!(bf16_num, bf16_x);
            }
        });
    }

    fn check_bf16_to_f32(num: bf16) {
        let simd_bf16 = u16x64::splat(num.to_bits());
        let simd_f32 = simd_bf16_to_f32(simd_bf16);
        simd_f32.to_array().iter().for_each(|&f32_x| {
            let f32_num = num.to_f32();
            if f32_num.is_nan() {
                assert!(f32_x.is_nan());
            } else {
                assert_eq!(f32_num, f32_x);
            }
        });
    }

    const BF16_VALUES: [bf16; 31] = [
        bf16::EPSILON,
        bf16::INFINITY,
        bf16::MAX,
        bf16::MIN,
        bf16::MIN_POSITIVE,
        bf16::NAN,
        bf16::NEG_INFINITY,
        bf16::MIN_POSITIVE_SUBNORMAL,
        bf16::MAX_SUBNORMAL,
        bf16::ONE,
        bf16::ZERO,
        bf16::NEG_ZERO,
        bf16::NEG_ONE,
        bf16::E,
        bf16::PI,
        bf16::FRAC_1_PI,
        bf16::FRAC_1_SQRT_2,
        bf16::FRAC_2_PI,
        bf16::FRAC_2_SQRT_PI,
        bf16::FRAC_PI_2,
        bf16::FRAC_PI_3,
        bf16::FRAC_PI_4,
        bf16::FRAC_PI_6,
        bf16::FRAC_PI_8,
        bf16::LN_10,
        bf16::LN_2,
        bf16::LOG10_E,
        bf16::LOG10_2,
        bf16::LOG2_E,
        bf16::LOG2_10,
        bf16::SQRT_2,
    ];

    #[test]
    fn test_simd_f32_to_bf16() {
        BF16_VALUES
            .iter()
            .for_each(|&x| check_f32_to_bf16(x.to_f32()));
    }

    #[test]
    fn test_simd_bf16_to_f32() {
        BF16_VALUES.iter().for_each(|&x| check_bf16_to_f32(x));
    }
}
