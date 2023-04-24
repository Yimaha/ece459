use rpkg::debversion;
use crate::Packages;
use crate::packages::Dependency;

impl Packages {
    /// Gets the dependencies of package_name, and prints out whether they are satisfied (and by which library/version) or not.
    pub fn deps_available(&self, package_name: &str) {
        if !self.package_exists(package_name) {
            println!("no such package {}", package_name);
            return;
        }
        println!("Package {}:", package_name);
        // println!("- dependency {:?}", "dep");
        // println!("+ {} satisfied by installed version {}", "dep", "459");

        // some sort of for loop...
        for dependency in self.dependencies.get(self.package_name_to_num.get(package_name).unwrap()).unwrap() {
            println!("- dependency {:?}", self.dep2str(dependency));
            match self.dep_is_satisfied(dependency) {
                None => {println!("-> not satisfied");},
                Some(package) => {
                    println!("+ {} satisfied by installed version {}", package, self.get_installed_debver(package).unwrap());
                }
            }
        }
    }

    /// Returns Some(package) which satisfies dependency dd, or None if not satisfied.
    pub fn dep_is_satisfied(&self, dd:&Dependency) -> Option<&str> {
        // presumably you should loop on dd
        for package in dd {
            if self.installed_debvers.contains_key(&package.package_num) {
                let dev_ver = self.installed_debvers.get(&package.package_num).unwrap();
                match &package.rel_version {
                    None => {
                        return Some(self.get_package_name(package.package_num))
                    },
                    Some(version) => {
                        let (op, v) = version;
                        if debversion::cmp_debversion_with_op(&op, dev_ver, &v.parse::<debversion::DebianVersionNum>().unwrap()) {
                            return Some(self.get_package_name(package.package_num))
                        }
                    }
                }
            }
        }
        return None;
    }

    /// Returns a Vec of packages which would satisfy dependency dd but for the version.
    /// Used by the how-to-install command, which calls compute_how_to_install().
    pub fn dep_satisfied_by_wrong_version(&self, dd:&Dependency) -> Vec<&str> {
        assert! (self.dep_is_satisfied(dd).is_none());
        let mut result = vec![];
        // another loop on dd
        for package in dd {
            if self.installed_debvers.contains_key(&package.package_num) {
                let dev_ver = self.installed_debvers.get(&package.package_num).unwrap();
                match &package.rel_version {
                    None => (),
                    Some(version) => {
                        let (op, v) = version;
                        if !debversion::cmp_debversion_with_op(&op, dev_ver, &v.parse::<debversion::DebianVersionNum>().unwrap()) {
                            result.push(self.get_package_name(package.package_num))
                        }
                    }
                }
            }
        }
        return result;
    }
}

