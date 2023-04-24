use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;
use rand::Rng;
use std::fs;
use std::sync::{Arc, Mutex};

pub struct Package {
    pub name: String,
}

pub struct PackageDownloader {
    pkg_start_idx: usize,
    num_pkgs: usize,
    event_sender: Sender<Event>,
    packages: Vec<String>,
}

impl PackageDownloader {
    pub fn new(pkg_start_idx: usize, num_pkgs: usize, event_sender: Sender<Event>) -> Self {
        Self {
            pkg_start_idx,
            num_pkgs,
            event_sender,
            packages: fs::read_to_string("data/packages.txt")
                .unwrap()
                .lines()
                .map(|p| p.to_owned())
                .collect(),
        }
    }

    pub fn run(&self, pkg_checksum: Arc<Mutex<Checksum>>) {
        let mut rng = rand::thread_rng();
        // Generate a set of packages and place them into the event queue
        // Update the package checksum with each package name\
        let rng_start = rng.gen_range(0..self.num_pkgs);
        for i in 0..self.num_pkgs {
            let name = self.packages[(self.pkg_start_idx + i) % self.packages.len()].to_owned();
            self.event_sender
                .send(Event::DownloadComplete(Package { name }))
                .unwrap();
            if i == rng_start {
                let mut guard = pkg_checksum.lock().unwrap();
                for i in 0..self.num_pkgs {
                    let name = &self.packages[(self.pkg_start_idx + i) % self.packages.len()];
                    guard.update(Checksum::with_sha256(&name));
                }
            }
        }

    }
}
