use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use regex::Regex;

use crate::Packages;
use crate::packages::RelVersionedPackageNum;

use rpkg::debversion;

const KEYVAL_REGEX : &str = r"(?P<key>(\w|-)+): (?P<value>.+)";
const PKGNAME_AND_VERSION_REGEX : &str = r"(?P<pkg>(\w|\.|\+|-)+)( \((?P<op>(<|=|>)(<|=|>)?) (?P<ver>.*)\))?";

impl Packages {
    /// Loads packages and version numbers from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate value into the installed_debvers map with the parsed version number.
    
    // pub fn print_example(&mut self) {
    //     println!("package example {}", self.available_debvers.get(self.package_name_to_num.get("libdbi1").unwrap()).unwrap());
    //     println!("package example {}", self.md5sums.get(self.package_name_to_num.get("libdbi1").unwrap()).unwrap());
    //     println!("package example {}", self.deps2str(self.dependencies.get(self.package_name_to_num.get("libdbi1").unwrap()).unwrap()));
    // }
    
    pub fn parse_installed(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        let mut package = 0;
        if let Ok(lines) = read_lines(filename) {
            for line in lines {
                if let Ok(ip) = line {
                    // do something with ip
                    match kv_regexp.captures(&ip) {
                        None => (),
                        Some(caps) => {
                            let (key, value) = (caps.name("key").unwrap().as_str(), caps.name("value").unwrap().as_str());
                            if key.eq("Package") {
                                package = self.get_package_num_inserting(&value);
                            } else if key.eq("Version") {
                                let debver = value.trim().parse::<debversion::DebianVersionNum>().unwrap();
                                self.installed_debvers.insert(package, debver);
                            }
                        }
                    }
                }
            }
        }
        println!("Packages installed: {}", self.installed_debvers.keys().len());
    }


    /// Loads packages, version numbers, dependencies, and md5sums from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate values into the dependencies, md5sum, and available_debvers maps.
    pub fn parse_packages(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        let pkgver_regexp = Regex::new(PKGNAME_AND_VERSION_REGEX).unwrap();
        let mut package = 0;

        if let Ok(lines) = read_lines(filename) {
            for line in lines {
                if let Ok(ip) = line {
                    // do more things with ip
                    match kv_regexp.captures(&ip) {
                        None => (),
                        Some(caps) => {
                            let (key, value) = (caps.name("key").unwrap().as_str(), caps.name("value").unwrap().as_str());
                            if key.eq("Package") {
                                package = self.get_package_num_inserting(&value);
                            } else if key.eq("Version") {
                                let debver = value.trim().parse::<debversion::DebianVersionNum>().unwrap();
                                self.available_debvers.insert(package, debver);
                            } else if key.eq("MD5sum") {
                                self.md5sums.insert(package, String::from(value.trim()));
                            } else if key.eq("Depends") {
                                // first split by ","
                                let mut result = vec![];
                                let split = value.trim().split(",");
                                for s in split {
                                    let raw_string = s.split("|");
                                    let mut depend: Vec<RelVersionedPackageNum> = vec![];
                                    for s in raw_string {
                                        match pkgver_regexp.captures(&s) {
                                            None => (),
                                            Some(caps) => {
                                                let pkg = caps.name("pkg").unwrap().as_str();
                                                let pac_num = self.get_package_num_inserting(&pkg);
                                                match caps.name("op") {
                                                    None => {
                                                        depend.push(RelVersionedPackageNum {
                                                            package_num: pac_num,
                                                            rel_version: None
                                                        })
                                                    },
                                                    Some(val) => {
                                                        let op = val.as_str();
                                                        let ver = caps.name("ver").unwrap().as_str();
                                                        depend.push(RelVersionedPackageNum {
                                                            package_num: pac_num,
                                                            rel_version: Some((op.parse::<debversion::VersionRelation>().unwrap(), ver.to_string()))
                                                        })
                                                    }
                                                }
                                                
                                            }
                                        }
                                    }
                                    result.push(depend);
                                }   
                                self.dependencies.insert(package, result);
                            }
                        }
                    }
                }
            }
        }
        println!("Packages available: {}", self.available_debvers.keys().len());
    }
}


// standard template code downloaded from the Internet somewhere
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
