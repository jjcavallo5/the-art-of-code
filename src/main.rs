// Operators: https://doc.rust-lang.org/std/ops/index.html

mod data;
mod micrograd;
mod network;

type Tensor = Vec<micrograd::Value>;

fn print_tensor(tensor: &Tensor) {
    for val in tensor {
        print!("{:.6} ", val.value());
    }
    println!()
}

fn cross_entropy(outs: &Tensor, label: &Tensor) -> micrograd::Value {
    let mut sum = micrograd::Value::new(0.0);
    for idx in 0..outs.len() {
        let loss_idx = &outs[idx] - &label[idx];
        let sqr_error = &loss_idx * &loss_idx;
        sum = &sum + &sqr_error;
    }

    let num_classes = micrograd::Value::new(label.len() as f64);
    return &sum / &num_classes;
}

fn softmax(logits: &Tensor) -> Tensor {
    let mut total = micrograd::Value::new(0.0);
    let max = logits
        .iter()
        .max_by(|a, b| a.value().total_cmp(&b.value()))
        .expect("NaN in logits");

    for idx in 0..logits.len() {
        let increment = &logits[idx] - max;
        total = &total + &increment.exp();
    }

    let mut out = Vec::new();
    for logit in logits {
        let normalized_lgt = logit - max;
        let exp_logit = normalized_lgt.exp();
        let sftmx_logit = &exp_logit / &total;
        out.push(sftmx_logit);
    }

    return out;
}

fn main() {
    data::get_files();
    let mut train_dataset = data::get_dataset(false);
    let mut test_dataset = data::get_dataset(true);

    let layer1 = network::LinearLayer::new(28 * 28, 256);
    let layer2 = network::LinearLayer::new(256, 10);

    const LR: f64 = 0.01;
    const LOG_EVERY: usize = 16;
    const BATCH_SIZE: usize = 16;
    let mut acc_loss = 0.0;
    let mut num_correct = 0;
    println!("Starting training...");
    for idx in 0..10_000 {
        if idx % LOG_EVERY == 0 {
            print!("[ {} / {} ]:  ", idx, train_dataset.len());
            print!("Loss: {}  ", acc_loss);
            print!("Acc: {} / {}", num_correct, LOG_EVERY);
            println!();
            acc_loss = 0.0;
            num_correct = 0;
        }
        let (img, lbl) = train_dataset.next();
        let l1 = layer1.forward(img);
        let l2: Tensor = l1.iter().map(|v| v.relu()).collect();
        let logits = layer2.forward(l2);
        let probs = softmax(&logits);
        let loss = cross_entropy(&probs, &lbl);
        acc_loss += loss.value();
        loss.backward();

        // Gradient accumulation for batch size replacement
        if idx % BATCH_SIZE == 0 {
            for p in layer1.parameters() {
                p.step(LR);
            }
            for p in layer2.parameters() {
                p.step(LR);
            }
        }

        let max_prob_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
            .expect("Max value not found")
            .0;
        let max_lbl_idx = lbl
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
            .expect("Max value not found")
            .0;
        if max_prob_idx == max_lbl_idx {
            num_correct += 1;
        }
    }

    let (img, _) = test_dataset.next();
    let l1 = layer1.forward(img);
    let outputs = layer2.forward(l1);
    print_tensor(&outputs);
}
