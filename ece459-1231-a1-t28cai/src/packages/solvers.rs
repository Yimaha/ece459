use crate::Packages;
use queues::*;
use linked_hash_set::LinkedHashSet;

impl Packages {
    /// Computes a solution for the transitive dependencies of package_name; when there is a choice A | B | C, 
    /// chooses the first option A. Returns a Vec<i32> of package numbers.
    ///
    /// Note: does not consider which packages are installed.
    pub fn transitive_dep_solution(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        let mut package_id = *self.get_package_num(package_name);
        // let deps : &Vec<Dependency> = &*self.dependencies.get(self.get_package_num(package_name)).unwrap();
        let mut dependency_set:LinkedHashSet<i32>  = LinkedHashSet::new();
        let mut frontier: Queue<i32> = queue![package_id];
        while frontier.size() != 0 {
            package_id = frontier.remove().unwrap();
            for dependency in self.dependencies.get(&package_id).unwrap() {
                // guarrenty to at least have 1 entry
                let target_id = dependency[0].package_num;
                if !dependency_set.contains(&target_id) {
                    frontier.add(target_id).unwrap();
                    dependency_set.insert(target_id);
                }
            }
        }

        // implement worklist

        return dependency_set.into_iter().collect();
    }

    /// Computes a set of packages that need to be installed to satisfy package_name's deps given the current installed packages.
    /// When a dependency A | B | C is unsatisfied, there are two possible cases:
    ///   (1) there are no versions of A, B, or C installed; pick the alternative with the highest version number (yes, compare apples and oranges).
    ///   (2) at least one of A, B, or C is installed (say A, B), but with the wrong version; of the installed packages (A, B), pick the one with the highest version number.
    pub fn compute_how_to_install(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        let mut package_id = *self.get_package_num(package_name);
        let mut dependency_set:LinkedHashSet<i32>  = LinkedHashSet::new();
        let mut frontier: Queue<i32> = queue![package_id];
        while frontier.size() != 0 {
            package_id = frontier.remove().unwrap();
            for dependencies in self.dependencies.get(&package_id).unwrap() {
                // guarrenty to at least have 1 entry
                if self.dep_is_satisfied(dependencies) != None { // no thing need to be done
                    continue;
                } 

                let mut ideal_upgrade_ver =  None; 
                let mut upgrad_id = -1;
                let wrong_version_list = self.dep_satisfied_by_wrong_version(dependencies);

                if wrong_version_list.len() != 0 { // we need to upgrade a dependency to new version
                    for package_name in wrong_version_list {
                        match ideal_upgrade_ver {
                            None => {
                                upgrad_id = *self.get_package_num(package_name);
                                ideal_upgrade_ver = Some(self.installed_debvers.get(&upgrad_id).unwrap());
                            },
                            Some(dvn) => { 
                                let potential_id =  *self.get_package_num(package_name);
                                let compare = self.installed_debvers.get(&potential_id).unwrap();
                                if dvn < compare {
                                    ideal_upgrade_ver = Some(compare);
                                    upgrad_id = potential_id;
                                }
                            }
                        }
                    }
                } else { // non dependency installed, trying to get a brand new dependency
                    for rvpn in dependencies {
                        let package_id = rvpn.package_num;
                        match ideal_upgrade_ver {
                            None => {
                                upgrad_id = package_id;
                                ideal_upgrade_ver = Some(self.available_debvers.get(&package_id).unwrap());
                            },
                            Some(dvn) => { 
                                let compare = self.available_debvers.get(&package_id).unwrap();
                                if dvn < compare {
                                    ideal_upgrade_ver = Some(compare);
                                    upgrad_id = package_id;
                                }
                            }
                        }
                    }
                }

                
                if !dependency_set.contains(&upgrad_id){
                    frontier.add(upgrad_id).unwrap();
                    dependency_set.insert(upgrad_id);
                } 
            }
        }


        return dependency_set.into_iter().collect();
    }
}
