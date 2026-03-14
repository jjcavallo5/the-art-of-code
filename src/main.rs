mod data;
mod micrograd;
mod network;

type Tensor = Vec<micrograd::Value>;

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

fn cross_entropy_loss(probs: &Tensor, label: &Tensor) -> micrograd::Value {
    let epsilon = micrograd::Value::new(1e-8);
    let mut loss = micrograd::Value::new(0.0);
    for idx in 0..probs.len() {
        if label[idx].value() > 0.5 {
            let log_prob = (&probs[idx] + &epsilon).log();
            loss = &loss - &log_prob;
        }
    }
    return loss;
}

fn main() {
    data::get_files();
    let mut train_dataset = data::get_dataset(false);

    let layer1 = network::LinearLayer::new(28 * 28, 32);
    let layer2 = network::LinearLayer::new(32, 10);

    const LR: f64 = 0.001;
    const LOG_EVERY: usize = 500;
    let train_size = train_dataset.len();
    let num_epochs = 10;

    // Load fixed training set
    println!("Loading {} training samples...", train_size);
    let mut train_imgs: Vec<Vec<f64>> = Vec::new();
    let mut train_lbls: Vec<Vec<f64>> = Vec::new();
    for _ in 0..train_size {
        let (img, lbl) = train_dataset.next();
        train_imgs.push(img.iter().map(|v| v.value()).collect());
        train_lbls.push(lbl.iter().map(|v| v.value()).collect());
    }

    println!(
        "Training (784->32->10, {} samples, {} epochs, lr={})...",
        train_size, num_epochs, LR
    );
    for epoch in 0..num_epochs {
        let mut acc_loss = 0.0;
        let mut num_correct = 0;
        for idx in 0..train_size {
            let img: Tensor = train_imgs[idx]
                .iter()
                .map(|v| micrograd::Value::new(*v))
                .collect();
            let lbl: Tensor = train_lbls[idx]
                .iter()
                .map(|v| micrograd::Value::new(*v))
                .collect();

            if idx % LOG_EVERY == 0 {
                println!(
                    "[EPOCH {}: {} / {}] Loss: {}",
                    epoch,
                    idx,
                    train_size,
                    acc_loss / idx as f64,
                )
            }

            let l1 = layer1.forward(img);
            let l2: Tensor = l1.iter().map(|v| v.relu()).collect();
            let logits = layer2.forward(l2);
            let probs = softmax(&logits);
            let loss = cross_entropy_loss(&probs, &lbl);
            acc_loss += loss.value();
            loss.backward();

            for p in layer1.parameters().0 {
                p.step(LR);
            }
            for p in layer1.parameters().1 {
                p.step(LR);
            }
            for p in layer2.parameters().0 {
                p.step(LR);
            }
            for p in layer2.parameters().1 {
                p.step(LR);
            }

            let pred = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
                .unwrap()
                .0;
            let target = train_lbls[idx]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            if pred == target {
                num_correct += 1;
            }
        }
        println!(
            "Epoch {:2}: Loss: {:.4}  Train Acc: {:.1}%",
            epoch,
            acc_loss / train_size as f64,
            num_correct as f64 / train_size as f64 * 100.0
        );
    }

    // Test evaluation
    let mut test_dataset = data::get_dataset(true);
    let mut test_correct = 0;
    let test_samples = test_dataset.len();
    for _ in 0..test_samples {
        let (img, lbl) = test_dataset.next();
        let img_vals: Tensor = img
            .iter()
            .map(|v| micrograd::Value::new(v.value()))
            .collect();
        let l1 = layer1.forward(img_vals);
        let l2: Tensor = l1.iter().map(|v| v.relu()).collect();
        let logits = layer2.forward(l2);
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
            .unwrap()
            .0;
        let target = lbl
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
            .unwrap()
            .0;
        if pred == target {
            test_correct += 1;
        }
    }
    println!(
        "\nTest accuracy: {} / {} ({:.1}%)",
        test_correct,
        test_samples,
        test_correct as f64 / test_samples as f64 * 100.0
    );
}
