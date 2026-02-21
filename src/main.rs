// Operators: https://doc.rust-lang.org/std/ops/index.html

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
}

impl Dataset {
    fn next(&mut self) -> ([u8; 28 * 28], u8) {
        let mut img = [0; 28 * 28];
        let mut lbl = [0];

        self.img_reader
            .read(&mut img)
            .expect("Failed to read image");
        self.lbl_reader
            .read(&mut lbl)
            .expect("Failed to read label");
        self.index += 1;

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

    // Skip past header
    let mut buffer = [0; 16];
    img_reader.read(&mut buffer).unwrap();

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
    };
}

fn print_sample(img: [u8; 28 * 28], lbl: u8) {
    for row in 0..28 {
        for col in 0..28 {
            let val = img[row * 28 + col];
            print!("{val} ");
        }
        println!();
    }
    println!("Label: {lbl}");
}

fn main() {
    get_files();
    let mut train_dataset = get_dataset(false);
    let mut test_dataset = get_dataset(true);
    let (img, label) = train_dataset.next();
    print_sample(img, label);
    let (img_2, label_2) = test_dataset.next();
    print_sample(img_2, label_2);
}
