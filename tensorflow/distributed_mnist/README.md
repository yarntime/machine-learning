## 执行命令


```
ps 命令：

python mnist.py --ps_hosts=192.168.254.44:2222 --worker_hosts=192.168.254.44:2224,192.168.254.45:2225 --job_name=ps --task_index=0



worker 命令:

python mnist.py --ps_hosts=192.168.254.44:2222 --worker_hosts=192.168.254.44:2224,192.168.254.45:2225 --job_name=worker --task_index=0

python mnist.py --ps_hosts=192.168.254.44:2222 --worker_hosts=192.168.254.44:2224,192.168.254.45:2225 --job_name=worker --task_index=1

```

存储的model在 trained_model 目录下， 

```
拷贝model到版本2目录下：
cp trained_model/* /tmp/monitored/2 -r
```


使用命令加载model到serving中使用：

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --enable_batching --port=9000 --model_name=mnist --model_base_path=/tmp/monitored/
```

测试训练好的模型：

```
bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
```