cargo run --release -- --raw-spark data/from_paper.log --to-parse "17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally" --before "split: hdfs://hostname/2kSOSP.log:29168+7292" --after "Found block" --cutoff 3 --original > out_original.txt
cargo run --release -- --raw-spark data/from_paper.log --to-parse "17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally" --before "split: hdfs://hostname/2kSOSP.log:29168+7292" --after "Found block" --cutoff 3 --single-map --num-threads=2 > out.txt
python3 compare.py out_original.txt out.txt
cargo run --release -- --raw-spark data/from_paper.log --to-parse "17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally" --before "split: hdfs://hostname/2kSOSP.log:29168+7292" --after "Found block" --cutoff 3 --num-threads=8 > out2.txt
python3 compare.py out_original.txt out2.txt


cargo run --quiet --release -- --raw-spark data/from_paper.log --to-parse "17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally" --before "split: hdfs://hostname/2kSOSP.log:29168+7292" --after "Found block" --cutoff 3 --original > out_original.txt
cargo run --quiet --release -- --raw-spark data/from_paper.log --to-parse "17/06/09 20:11:11 INFO storage.BlockManager: Found block rdd_42_20 locally" --before "split: hdfs://hostname/2kSOSP.log:29168+7292" --after "Found block" --cutoff 3 --single-map --num-threads=8 > out.txt
python3 compare.py out_original.txt out.txt

cargo run --quiet --release -- --raw-linux data/Linux_2k.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --original > out_original.txt
cargo run --quiet --release -- --raw-linux data/Linux_2k.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --single-map --num-threads=8 > out.txt
python3 compare.py out_original.txt out.txt
cargo run --quiet --release -- --raw-linux data/Linux_2k.log --to-parse "Jun 23 23:30:05 combo sshd(pam_unix)[26190]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.22.3.51  user=root" --before "rhost=<*> user=root" --after "session opened" --cutoff 100 --num-threads=8> out2.txt
python3 compare.py out_original.txt out2.txt

time cargo run --release -- --raw-healthapp data/HealthApp.log --to-parse "20171223-22:15:41:672|Step_StandReportReceiver|30002312|REPORT : 7028 5017 150539 240" --before "calculateAltitudeWithCache totalAltitude=240" --after "onStandStepChanged 3601" --cutoff 10 --original > out_original.txt
time cargo run --release -- --raw-healthapp data/HealthApp.log --to-parse "20171223-22:15:41:672|Step_StandReportReceiver|30002312|REPORT : 7028 5017 150539 240" --before "calculateAltitudeWithCache totalAltitude=240" --after "onStandStepChanged 3601" --cutoff 10 --single-map --num-threads=8 > out.txt
python3 compare.py out_original.txt out.txt
time cargo run --release -- --raw-healthapp data/HealthApp.log --to-parse "20171223-22:15:41:672|Step_StandReportReceiver|30002312|REPORT : 7028 5017 150539 240" --before "calculateAltitudeWithCache totalAltitude=240" --after "onStandStepChanged 3601" --cutoff 10 --num-threads=8> out2.txt
python3 compare.py out_original.txt out2.txt

time cargo run --release -- --raw-hpc data/HPC.log --to-parse "58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176" --before-line "58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177" --after-line "58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175" --cutoff 106 --original > out_original.txt
time cargo run --release -- --raw-hpc data/HPC.log --to-parse "58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176" --before-line "58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177" --after-line "58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175" --cutoff 106 --single-map --num-threads=8 > out.txt
python3 compare.py out_original.txt out.txt
time cargo run --release -- --raw-hpc data/HPC.log --to-parse "58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176" --before-line "58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177" --after-line "58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175" --cutoff 106 --num-threads=8> out2.txt
python3 compare.py out_original.txt out2.txt