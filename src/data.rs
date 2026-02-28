use std::fs;
use std::io;
use std::io::*;
use std::process;

use crate::micrograd;

pub fn get_files() {
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

        std::process::Command::new("curl")
            .args([&format!("{url}"), "--output", &format!("{file}.gz")])
            .output()
            .expect("Failed to fetch data");

        process::Command::new("gzip")
            .args(["-d", &format!("{file}.gz")])
            .output()
            .expect("Failed to decompress data");
    }
}

pub struct Dataset {
    img_reader: io::BufReader<fs::File>,
    lbl_reader: io::BufReader<fs::File>,
    index: u32,
    len: u32,
}

impl Dataset {
    pub fn next(&mut self) -> (Vec<micrograd::Value>, Vec<micrograd::Value>) {
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

    pub fn len(&self) -> u32 {
        return self.len;
    }
}

pub fn get_dataset(is_test: bool) -> Dataset {
    let mut split = "train";
    if is_test {
        split = "test";
    }

    // Open train image file
    let fp_images = fs::File::open(format!("{split}_images")).unwrap();
    let mut img_reader = io::BufReader::new(fp_images);

    // Get header
    let mut buffer = [0; 16];
    img_reader.read(&mut buffer).unwrap();

    // Get dataset length
    let length = buffer[7] as u32 + buffer[6] as u32 * 256;

    // Open train label file
    let fp_labels = fs::File::open(format!("{split}_labels")).unwrap();
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
