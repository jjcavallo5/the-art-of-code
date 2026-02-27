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

type Tensor = Vec<micrograd::Value>;

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
    fn next(&mut self) -> (Vec<micrograd::Value>, Vec<micrograd::Value>) {
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
        let mut img = Vec::new();
        for idx in 0..img_u8.len() {
            let val = micrograd::Value::new(img_u8[idx] as f64 / 255.0);
            img.push(val);
        }

        let mut label = Vec::new();
        // One-hot encode label
        for idx in 0..10 {
            if lbl[0] == idx {
                label.push(micrograd::Value::new(1.0));
            } else {
                label.push(micrograd::Value::new(0.0));
            }
        }

        return (img, label);
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

fn cross_entropy(outs: Tensor, label: Tensor) -> micrograd::Value {
    let mut sum = micrograd::Value::new(0.0);
    for idx in 0..outs.len() {
        let loss_idx = &outs[idx] - &label[idx];
        let sqr_error = &loss_idx * &loss_idx;
        sum = &sum + &sqr_error;
    }

    let num_classes = micrograd::Value::new(label.len() as f64);
    return &sum / &num_classes;
}

fn main() {
    get_files();
    let mut train_dataset = get_dataset(false);
    let mut test_dataset = get_dataset(true);
    // print_sample(img, lbl);

    // let a = micrograd::Value::new(2.0);
    // let b = micrograd::Value::new(3.0);
    // let c = &a * &b;
    // let L = &c + &a;
    // L.backward();
    let layer = network::LinearLayer::new(28 * 28, 10);

    for idx in 0..train_dataset.len {
        if idx % 100 == 0 {
            println!("{} / {}", idx, train_dataset.len);
        }
        let (img, lbl) = train_dataset.next();
        let output = layer.forward(img);
        let loss = cross_entropy(output, lbl);
        loss.backward();
    }

    let (img, _) = test_dataset.next();
    let outputs = layer.forward(img);
    for out in outputs {
        println!("{:?}", out.value())
    }
}
