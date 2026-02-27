use std::cell::RefCell;
use std::collections::HashSet;
use std::ops;
use std::rc::Rc;

// Gives us a mutatable reference
// that can have multiple owners
type ValueRef = Rc<RefCell<Node>>;

// Node struct for storing data
#[derive(Debug)]
struct Node {
    value: f64,
    grad: f64,
    local_grads: Vec<f64>,
    children: Vec<ValueRef>,
}

#[derive(Debug, Clone)]
pub struct Value {
    node: ValueRef,
}

impl ops::Add<&Value> for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        let out = Value::new(self.node.borrow().value + other.node.borrow().value);
        out.node.borrow_mut().children.push(self.node.clone());
        out.node.borrow_mut().children.push(other.node.clone());
        out.node.borrow_mut().local_grads.push(1.0);
        out.node.borrow_mut().local_grads.push(1.0);
        return out;
    }
}

impl ops::Sub<&Value> for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        let out = Value::new(self.node.borrow().value - other.node.borrow().value);
        out.node.borrow_mut().children.push(self.node.clone());
        out.node.borrow_mut().children.push(other.node.clone());
        out.node.borrow_mut().local_grads.push(1.0);
        out.node.borrow_mut().local_grads.push(1.0);
        return out;
    }
}

impl ops::Mul<&Value> for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        let out = Value::new(self.node.borrow().value * other.node.borrow().value);
        let other_val = other.node.borrow().value;
        let self_val = self.node.borrow().value;
        out.node.borrow_mut().children.push(self.node.clone());
        out.node.borrow_mut().children.push(other.node.clone());
        out.node.borrow_mut().local_grads.push(other_val);
        out.node.borrow_mut().local_grads.push(self_val);
        return out;
    }
}

impl ops::Div<&Value> for &Value {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        let other_inv = other.pow(-1.0);
        return self * &other_inv;
    }
}

impl Value {
    pub fn new(value: f64) -> Self {
        return Self {
            node: Rc::new(RefCell::new(Node {
                value,
                grad: 0.0,
                local_grads: Vec::new(),
                children: Vec::new(),
            })),
        };
    }

    pub fn backward(self) {
        let topo = build_topo(self.node.clone());

        self.node.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            for (idx, child) in v.borrow().children.iter().enumerate() {
                child.borrow_mut().grad += v.borrow().local_grads[idx];
            }
        }
    }

    pub fn grad(&self) -> f64 {
        return self.node.borrow().grad;
    }
    pub fn value(&self) -> f64 {
        return self.node.borrow().value;
    }
    pub fn pow(&self, pow: f64) -> Value {
        let out = Value::new(self.node.borrow().value.powf(pow));
        let self_val = self.node.borrow().value;
        out.node.borrow_mut().children.push(self.node.clone());
        out.node
            .borrow_mut()
            .local_grads
            .push(pow * self_val.powf(pow - 1.));
        return out;
    }
    pub fn relu(&self) -> Value {
        let self_val = self.node.borrow().value;
        if self_val < 0.0 {
            let out = Value::new(0.0);
            out.node.borrow_mut().children.push(self.node.clone());
            out.node.borrow_mut().local_grads.push(0.0);
            return out;
        } else {
            let out = Value::new(self_val);
            out.node.borrow_mut().children.push(self.node.clone());
            out.node.borrow_mut().local_grads.push(1.0);
            return out;
        }
    }
    pub fn exp(&self) -> Value {
        let self_val = self.node.borrow().value;
        let out_val = 2.718281828459_f64.powf(self_val);
        let out = Value::new(out_val);
        out.node.borrow_mut().children.push(self.node.clone());
        out.node.borrow_mut().local_grads.push(out_val);
        return out;
    }
    pub fn step(&self, learning_rate: f64) {
        let mut node = self.node.borrow_mut();
        node.value -= node.grad * learning_rate;
        node.grad = 0.0;
    }
}

fn build_topo(root: ValueRef) -> Vec<ValueRef> {
    let mut visited = HashSet::new();
    let mut topo = Vec::new();

    fn dfs(v: ValueRef, topo: &mut Vec<ValueRef>, visited: &mut HashSet<usize>) {
        let value_ptr = Rc::as_ptr(&v) as usize;
        if !visited.contains(&value_ptr) {
            visited.insert(value_ptr);
            for child in &v.borrow().children {
                dfs(child.clone(), topo, visited)
            }
            topo.push(v)
        }
    }

    dfs(root, &mut topo, &mut visited);
    return topo;
}
