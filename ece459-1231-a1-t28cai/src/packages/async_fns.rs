use curl::Error;
use curl::easy::{Easy2, Handler, WriteError};
use curl::multi::{Easy2Handle, Multi};
use std::time::Duration;
use std::str;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::Packages;

pub struct Collector(Box<String>);
impl Handler for Collector {
    fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
        (*self.0).push_str(str::from_utf8(&data.to_vec()).unwrap());
        Ok(data.len())
    }
}

const DEFAULT_SERVER : &str = "ece459.patricklam.ca:4590";
impl Drop for Packages {
    fn drop(&mut self) {
        self.execute()
    }
}

static EASYKEY_COUNTER: AtomicI32 = AtomicI32::new(0);

pub struct AsyncState {
    server : String,
    easys : Vec<Easy2Handle<Collector>>,
    multi : Multi,
    requests: Vec<(String, String)> // package name and package version
}

impl AsyncState {
    pub fn new() -> AsyncState {
        AsyncState {
            server : String::from(DEFAULT_SERVER),
            easys : Vec::new(),
            multi : Multi::new(),
            requests: Vec::new()
        }

    }
}

impl Packages {
    pub fn set_server(&mut self, new_server:&str) {
        self.async_state.server = String::from(new_server);
        self.async_state.multi.pipelining(true, true).unwrap();
    }

    /// Retrieves the version number of pkg and calls enq_verify_with_version with that version number.
    pub fn enq_verify(&mut self, pkg:&str) {
        let version = self.get_available_debver(pkg);
        match version {
            None => { println!("Error: package {} not defined.", pkg); return },
            Some(v) => { 
                let vs = &v.to_string();
                self.enq_verify_with_version(pkg, vs); 
            }
        };
    }

    pub fn init(&self, multi:&Multi, url:&str) -> Result<Easy2Handle<Collector>, Error> {
        let mut easy = Easy2::new(Collector(Box::new(String::new())));
        easy.url(url)?;
        easy.verbose(false)?;
        Ok(multi.add2(easy).unwrap())
    }
    
    /// Enqueues a request for the provided version/package information. Stores any needed state to async_state so that execute() can handle the results and print out needed output.
    pub fn enq_verify_with_version(&mut self, pkg:&str, version:&str) {
        let url = format!("{}/rest/v1/checksums/{}/{}", self.async_state.server, pkg, version);
        println!("queueing request http://{}", url);
        self.async_state.easys.push(self.init(&self.async_state.multi, &url).unwrap());
        self.async_state.requests.push((String::from(pkg), String::from(version)))
    }

    /// Asks curl to perform all enqueued requests. For requests that succeed with response code 200, compares received MD5sum with local MD5sum (perhaps stored earlier). For requests that fail with 400+, prints error message.
    pub fn execute(&mut self) {
        while self.async_state.multi.perform().unwrap() > 0 {
            // .messages() may have info for us here...
            self.async_state.multi.wait(&mut [], Duration::from_secs(10)).unwrap();
        }

        for eh in self.async_state.easys.drain(..) {
            let num: usize = EASYKEY_COUNTER.load(Ordering::SeqCst) as usize;
            let (pkg, version) = &self.async_state.requests[num];
            let mut handler_after:Easy2<Collector> = self.async_state.multi.remove2(eh).unwrap();
            let code = handler_after.response_code().unwrap();
            if code != 200 {
                println!("got error {} on request for package {} version {}", code, pkg, version)
            } else {
                let hash: String = String::clone(&*handler_after.get_ref().0);
                let stored: &str = self.md5sums.get(self.package_name_to_num.get(pkg).unwrap()).unwrap();
                println!("verifying {}, matches: {:?}", pkg,  stored.eq(hash.as_str()))
            }
            EASYKEY_COUNTER.fetch_add(1, Ordering::SeqCst);
        }
        
    }
}
