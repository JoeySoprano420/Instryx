//! String manipulation utilities

pub type String = std::string::String;

impl StringExt for String {
    fn format(&self, args: &[&str]) -> String {
        // TODO: Implement string formatting
        format!("{} - formatted with {} args", self, args.len())
    }
}

pub trait StringExt {
    fn format(&self, args: &[&str]) -> String;
}