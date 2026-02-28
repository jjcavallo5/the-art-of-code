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

fn softmax(logits: Tensor) -> Tensor {
    let mut total = micrograd::Value::new(0.0);
    for idx in 0..logits.len() {
        total = &total + &logits[idx].exp(); //2.718281828459_f64.powf(logits[idx].value());
    }

    let mut out = Vec::new();
    for logit in logits {
        let exp_logit = logit.exp();
        let sftmx_logit = &exp_logit / &total;
        out.push(sftmx_logit);
    }

    return out;
}

fn main() {
    data::get_files();
    let mut train_dataset = data::get_dataset(false);
    let mut test_dataset = data::get_dataset(true);
    // print_sample(img, lbl);

    // let a = micrograd::Value::new(2.0);
    // let b = micrograd::Value::new(3.0);
    // let c = &a * &b;
    // let L = &c + &a;
    // L.backward();
    let layer = network::LinearLayer::new(28 * 28, 10);

    const LR: f64 = 0.01;
    let mut acc_loss = 0.0;
    for idx in 0..10_000 {
        if idx % 100 == 0 {
            println!("[ {} / {} ]:\tLoss: {}", idx, train_dataset.len(), acc_loss);
            acc_loss = 0.0;
        }
        let (img, lbl) = train_dataset.next();
        let l1 = layer.forward(img);
        let l2 = l1.iter().map(|v| v.relu()).collect();
        let probs = softmax(l2);
        // print_tensor(&probs);
        let loss = cross_entropy(probs, lbl);
        acc_loss += loss.value();
        loss.backward();

        for p in layer.parameters() {
            p.step(LR);
        }
    }

    let (img, _) = test_dataset.next();
    let outputs = layer.forward(img);
    print_tensor(&outputs);
}
