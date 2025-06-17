use std::time::{Duration, Instant};

pub struct Timer {
    name: String,
    begin: Option<Instant>,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            begin: Some(Instant::now()),
        }
    }

    /// Create a new timer with the same start time as this timer.
    pub fn copy(&self, name: &str) -> Self {
        Self {
            name: name.to_owned(),
            begin: self.begin,
        }
    }

    pub fn report_and_reset(&mut self, name: &str) -> Duration {
        let (elapsed, now) = self.report();
        self.name = name.to_owned();
        self.begin = Some(now);
        elapsed
    }

    fn report(&self) -> (Duration, Instant) {
        let now = Instant::now();
        let delta = now - self.begin.unwrap();
        println!("{}: {:?}", self.name, delta);
        (delta, now)
    }
}

impl Timer {
    pub fn end(mut self) -> Duration {
        let elapsed = self.report().0;
        self.begin = None;
        elapsed
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if self.begin.is_some() {
            self.report();
        }
    }
}
