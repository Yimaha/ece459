# Summary
During the creation of the maps and token list, by NOT enabling --single-map, follow by num-threads indicating how many thread is used (defaulty 1). it essentially create a list of thread executing on 1/nth of the text file given to the code, and convert them together at the end.

# Tech details:
Similar to a map reduce job, my objective is to ensure the result of the sequential (original) implementaion and the parallel implementation should be identical, though the order of the map might not be the same. This means we hand 1 extra line before and 1 extra line after for each task when necessary. this allow a complete match of the map as if we execute them linearly.

We utilzed the DashMap crate (https://github.com/xacrimon/dashmap) which allow async reads/write on teh same map. on average, the concurrent-hashmap implementation is about 40 - 50% faster than the sequential implementation, which is quiet nice. 

It is worth noting that this approach has a notable downside which is all the thread are accessing the same map at the same time. this means, there will be a significant amount of time where certain thread might be blocked on awaiting access to the map. However, our test didn't have a big enough dataset (all within ranges of mb not even gb) and thus it is hard to see the difference. In general, it has almost identical performance when compared to the seperate-maps approach.

# Correctness:
I wrote a script that manually test through couple examples given in the readme follow py the compare.py provided by the TA, with 0 difference tolerence. by keeping the --original which enables the sequential implementation of hte program, I was able to make sure it passes majority of cases

# Speed:
We used the "time" command in bash to measure the speed of the program, for example
```
Files match within specified tolerance
    Finished release [optimized] target(s) in 0.03s
     Running `target/release/logram --raw-hpc data/HPC.log --to-parse '58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176' --before-line '58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177' --after-line '58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175' --cutoff 106 --original`

real    0m2.123s
user    0m2.076s
sys     0m0.016s
    Finished release [optimized] target(s) in 0.02s
     Running `target/release/logram --raw-hpc data/HPC.log --to-parse '58717 2185 boot_cmd new 1076865186 1 Targeting domains:node-D1 and nodes:node-[40-63] child of command 2176' --before-line '58728 2187 boot_cmd new 1076865197 1 Targeting domains:node-D2 and nodes:node-[72-95] child of command 2177' --after-line '58707 2184 boot_cmd new 1076865175 1 Targeting domains:node-D0 and nodes:node-[0-7] child of command 2175' --cutoff 106 --num-threads=8`

real    0m1.037s
user    0m4.073s
sys     0m1.185s
Files match within specified tolerance
```

first execution is the original runtime, measured to about 2.123s in real time, while the 2nd result is seperate map approach, and cost only about 1.037 sec. Note that double amount of time is spent in user-space, since we are creating multiple threads. do note that I tried to speed up the original solution by replacing the list with hashset. (it might took a lot longer if not )