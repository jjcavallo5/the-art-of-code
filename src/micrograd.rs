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
        let out = Value::new(self.node.borrow().value / other.node.borrow().value);
        let other_val = other.node.borrow().value;
        let self_val = self.node.borrow().value;
        out.node.borrow_mut().children.push(self.node.clone());
        out.node.borrow_mut().children.push(other.node.clone());

        let d_self = 1.0 / other_val;
        let d_other = -self_val / other_val.powf(2.0);
        out.node.borrow_mut().local_grads.push(d_self);
        out.node.borrow_mut().local_grads.push(d_other);
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
        let out_val = self.node.borrow().value.exp();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_backprop() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let result = &a + &b;
        result.backward(); // a.grad == b.grad == 1
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn mul_backprop() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let result = &a * &b;
        result.backward(); // a.grad == b, b.grad == a
        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn pow_backprop() {
        let a = Value::new(4.0);
        a.pow(2.0).backward();
        assert_eq!(a.grad(), 8.0);

        let b = Value::new(4.0);
        b.pow(-1.0).backward();
        assert_eq!(b.grad(), -0.0625);
    }

    #[test]
    fn div_backprop() {
        let a = Value::new(4.0);
        let b = Value::new(2.0);
        let result = &a / &b;
        result.backward();
        // f'g + g'x / g**2 => a.grad = (1*2 + 1*4) / 4 = 1.5
        // f'g + g'x / g**2 => b.grad = (1*4 + 1*2) / 16 = 0.375
        assert_eq!(a.grad(), 0.5);
        assert_eq!(b.grad(), -1.0);
    }

    #[test]
    fn exp_backprop() {
        let a = Value::new(2.0);
        a.exp().backward(); // a.grad == a.exp()
        assert_eq!(a.grad(), 2.0_f64.exp());
    }

    #[test]
    fn relu_backprop() {
        let a = Value::new(2.0);
        let b = Value::new(-2.0);
        a.relu().backward(); // a.grad = 1.0
        b.relu().backward(); // b.grad = 0.0
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 0.0);
    }

    #[test]
    fn test_step() {
        let a = Value::new(3.0);
        let b = Value::new(2.0);
        let c = &a * &b; // a.grad = 2, b.grad = 3
        c.backward();
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 3.0);

        a.step(0.1); // update by 2.0*0.1=0.2 => 2.8 new val
        b.step(0.1); // update by 3.0*0.1=0.3 => 1.7 new val

        assert_eq!(a.value(), 2.8);
        assert_eq!(b.value(), 1.7);
        assert_eq!(a.grad(), 0.0); // Gradients zeroed
        assert_eq!(b.grad(), 0.0); // Gradients zeroed
    }
}
