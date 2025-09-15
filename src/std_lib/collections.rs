//! Collections module

pub struct Vec<T> {
    data: std::vec::Vec<T>,
}

impl<T> Vec<T> {
    pub fn new() -> Self {
        Vec { data: std::vec::Vec::new() }
    }
    
    pub fn push(&mut self, item: T) {
        self.data.push(item);
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub type HashMap<K, V> = std::collections::HashMap<K, V>;
pub type HashSet<T> = std::collections::HashSet<T>;