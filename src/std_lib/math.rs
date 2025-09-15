//! Mathematical functions and constants

pub const PI: f64 = std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;

pub fn abs(x: f64) -> f64 {
    x.abs()
}

pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

pub fn sin(x: f64) -> f64 {
    x.sin()
}

pub fn cos(x: f64) -> f64 {
    x.cos()
}

pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

pub fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}