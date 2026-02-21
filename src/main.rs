// Operators: https://doc.rust-lang.org/std/ops/index.html

use std::mem;
use std::{
    fs::{self, File},
    io,
    io::prelude::*,
    process::Command,
};

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
            println!("File {file} exists. Skipping...");
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

fn read_train_images() -> io::Result<()> {
    let f = File::open("train_images").unwrap();
    let mut reader = io::BufReader::new(f);
    let mut buffer = [0; 16];
    reader.read(&mut buffer).unwrap(); // Header, don't care

    let mut first_image = [0; 28 * 28];
    reader.read(&mut first_image).unwrap();

    for row in 0..28 {
        for col in 0..28 {
            let val = first_image[row * 28 + col];
            print!("{val} ");
        }
        println!();
    }

    Ok(())
}

fn read_train_labels() {
    let f = File::open("train_labels").unwrap();
    let mut reader = io::BufReader::new(f);
    let mut buffer = [0; 8];
    reader.read(&mut buffer).unwrap(); // Header, don't care

    let mut first_label = [0; 1];
    reader.read(&mut first_label).unwrap();

    let lbl = first_label[0];
    println!("Label: {lbl}")
}

fn main() {
    println!("Hello world!");
    get_files();
    read_train_images();
    read_train_labels();
}
