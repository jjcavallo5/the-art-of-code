// Operators: https://doc.rust-lang.org/std/ops/index.html

mod micrograd;
mod network;

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
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
    // get_files();
    // let mut train_dataset = get_dataset(false);
    // let mut test_dataset = get_dataset(true);
    // let (img, lbl) = train_dataset.next();
    // print_sample(img, lbl);

    // let a = micrograd::Value::new(2.0);
    // let b = micrograd::Value::new(3.0);
    // let c = &a * &b;
    // let L = &c + &a;
    // L.backward();
    //
    let layer = network::LinearLayer::new(2, 3);
    let input = vec![micrograd::Value::new(0.0), micrograd::Value::new(1.0)];
    let output = layer.forward(input);
    let output_vals = output.iter().map(|x| x.value());
    for val in output_vals {
        println!("{:?}", val)
    }
}
