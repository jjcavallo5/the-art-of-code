// Operators: https://doc.rust-lang.org/std/ops/index.html

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    fs::{self, File},
    io,
    io::prelude::*,
    process::Command,
};

fn random() -> u64 {
    let hasher = RandomState::new().build_hasher();
    return hasher.finish();
}

fn get_files() {
    let urls = [
        "https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz",
        "https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz",
        "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz",
        "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz",
    ];
    let files = ["train_images", "train_labels", "test_images", "test_labels"];

    for idx in 0..urls.len() {
        let url = urls[idx];
        let file = files[idx];

        if fs::exists(file).expect("Failed to check file") {
            continue;
        }

        Command::new("curl")
            .args([&format!("{url}"), "--output", &format!("{file}.gz")])
            .output()
            .expect("Failed to fetch data");

        Command::new("gzip")
            .args(["-d", &format!("{file}.gz")])
            .output()
            .expect("Failed to decompress data");
    }
}

struct Dataset {
    img_reader: io::BufReader<File>,
    lbl_reader: io::BufReader<File>,
    index: u32,
    len: u32,
}

impl Dataset {
    fn next(&mut self) -> ([f32; 28 * 28], u8) {
        let mut img_u8 = [0; 28 * 28];
        let mut lbl = [0];

        self.img_reader
            .read(&mut img_u8)
            .expect("Failed to read image");
        self.lbl_reader
            .read(&mut lbl)
            .expect("Failed to read label");
        self.index += 1;

        // Normalize image to [0, 1]
        let mut img = [0_f32; 28 * 28];
        for idx in 0..img_u8.len() {
            img[idx] = img_u8[idx] as f32 / 255.0;
        }

        return (img, lbl[0]);
    }
}

fn get_dataset(is_test: bool) -> Dataset {
    let mut split = "train";
    if is_test {
        split = "test";
    }

    // Open train image file
    let fp_images = File::open(format!("{split}_images")).unwrap();
    let mut img_reader = io::BufReader::new(fp_images);

    // Get header
    let mut buffer = [0; 16];
    img_reader.read(&mut buffer).unwrap();

    // Get dataset length
    let length = buffer[7] as u32 + buffer[6] as u32 * 256;

    // Open train label file
    let fp_labels = File::open(format!("{split}_labels")).unwrap();
    let mut lbl_reader = io::BufReader::new(fp_labels);

    // Skip past header
    let mut buffer = [0; 8];
    lbl_reader.read(&mut buffer).unwrap(); // Header, don't care

    return Dataset {
        img_reader,
        lbl_reader,
        index: 0,
        len: length,
    };
}

#[derive(Debug, Clone)]
struct Value {
    id: u64,
    value: f32,
    grad: f32,
    children: Vec<Value>,
    local_grads: Vec<f32>,
}

impl Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        return Self {
            id: random(),
            value: self.value + other.value,
            grad: 0.0,
            children: vec![self, other],
            local_grads: vec![1.0, 1.0],
        };
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        return Self {
            id: random(),
            value: self.value * other.value,
            grad: 0.0,
            local_grads: vec![other.value, self.value],
            children: vec![self, other],
        };
    }
}

impl Value {
    fn new(value: f32) -> Self {
        return Self {
            id: random(),
            value,
            grad: 0.0,
            local_grads: Vec::new(),
            children: Vec::new(),
        };
    }

    fn backward(mut self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited_ids: Vec<u64> = Vec::new();
        self.grad = 1.0;
        build_topo(&mut topo, &mut visited_ids, self.clone());
        topo.reverse();

        for top in topo.clone() {
            println!("{}:{}", top.value, top.grad);
        }

        for mut v in topo {
            for idx in 0..v.children.len() {
                println!("grads: {}, {}", v.local_grads[idx], v.grad);
                v.children[idx].grad += v.local_grads[idx] * v.grad;
            }
        }
    }
}

fn build_topo(topo: &mut Vec<Value>, visited_ids: &mut Vec<u64>, v: Value) {
    if !visited_ids.contains(&v.id) {
        visited_ids.push(v.id);
        for child in &v.children {
            build_topo(topo, visited_ids, child.clone());
        }
        topo.push(v);
    }
}

fn print_sample(img: [f32; 28 * 28], lbl: u8) {
    for row in 0..28 {
        for col in 0..28 {
            print!("{} ", img[row * 28 + col]);
        }
        println!();
    }
    println!("Label: {lbl}");
}

fn main() {
    get_files();
    let mut train_dataset = get_dataset(false);
    let mut test_dataset = get_dataset(true);
    let (img, lbl) = train_dataset.next();
    print_sample(img, lbl);

    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = a.clone() * b.clone();
    let L = c.clone() + a.clone();
    L.backward();
    println!("grad: {}, val: {}", a.grad, c.grad);
}
