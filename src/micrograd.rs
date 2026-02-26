use std::cell::RefCell;
use std::collections::HashSet;
use std::ops;
use std::rc::Rc;

// Gives us a mutatable reference
// that can have multiple owners
type ValueRef = Rc<RefCell<Node>>;

// Node struct for storing data
struct Node {
    value: f64,
    grad: f64,
    local_grads: Vec<f64>,
    children: Vec<ValueRef>,
}

#[derive(Clone)]
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
