use super::{checksum::Checksum, idea::Idea, package::Package, Event};
use crossbeam::channel::{Receiver, Sender};
use rand::Rng;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct Student {
    id: usize,
    idea: VecDeque<Idea>,
    pkgs: VecDeque<Package>,
    event_sender: Sender<Event>,
    event_recv: Receiver<Event>,
    rng_counter: usize,
}

impl Student {
    pub fn new(id: usize, event_sender: Sender<Event>, event_recv: Receiver<Event>) -> Self {
        Self {
            id,
            event_sender,
            event_recv,
            idea: VecDeque::new(),
            pkgs: VecDeque::new(),
            rng_counter: 0,
        }
    }

    fn can_build_idea(&mut self) -> Option<&Idea> {
        match self.idea.front() {
            Some(idea) => {
                if self.pkgs.len() >= idea.num_pkg_required {
                    return Some(&idea);
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn build_idea(
        &mut self,
        idea_checksum: &Arc<Mutex<Checksum>>,
        pkg_checksum: &Arc<Mutex<Checksum>>,
    ) {
        while let Some(ref idea) = self.can_build_idea() {
            // Can only build ideas if we have acquired sufficient packages
            let pkgs_required = idea.num_pkg_required;
            {
                let mut idea_checksum = idea_checksum.lock().unwrap();
                idea_checksum.update(Checksum::with_sha256(&idea.name));
            }
            {
                let mut pkg_checksum = pkg_checksum.lock().unwrap();
                for _ in 0..pkgs_required {
                    if let Some(pkg) = self.pkgs.pop_front() {
                        pkg_checksum.update(Checksum::with_sha256(&pkg.name));
                    }
                }
            }
            self.idea.pop_front();
        }
    }

    pub fn run(&mut self, idea_checksum: Arc<Mutex<Checksum>>, pkg_checksum: Arc<Mutex<Checksum>>) {
        loop {

            let event = self.event_recv.recv().unwrap();
            let mut rng = rand::thread_rng();
            let rng_limit = rng.gen_range(1..50);
            let rng_key = rng.gen_range(0..rng_limit);
            match event {
                Event::NewIdea(idea) => {
                    self.idea.push_back(idea);
                    if rng_key == (self.rng_counter % 5) {
                        self.build_idea(&idea_checksum, &pkg_checksum);
                    }
                    self.rng_counter += 1;
                }

                Event::DownloadComplete(pkg) => {
                    // Getting a new package means the current idea may now be buildable, so the
                    // student attempts to build it
                    self.pkgs.push_back(pkg);
                    if rng_key == (self.rng_counter % 5) {
                        self.build_idea(&idea_checksum, &pkg_checksum);
                    }
                }

                Event::OutOfIdeas => {
                    self.build_idea(&idea_checksum, &pkg_checksum);
                    if !self.idea.is_empty() {
                        self.event_sender.send(Event::OutOfIdeas).unwrap();
                    } else {
                        // Any unused packages are returned to the queue upon termination
                        for pkg in self.pkgs.drain(..) {
                            self.event_sender
                                .send(Event::DownloadComplete(pkg))
                                .unwrap();
                        }
                        return;
                    }
                }
            }
        }
    }
}
