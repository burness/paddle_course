import paddle.v2 as paddle
import numpy as np

paddle.init(use_gpu=False)

x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
cost = paddle.layer.square_error_cost(input=y_predict, label=y)

parameters = paddle.parameters.create(cost)
optimizer = paddle.optimizer.Momentum(momentum=0)
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)
    # product model every 10 pass
    if isinstance(event, paddle.event.EndPass):
        if event.pass_id % 10 == 0:
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

def train_reader():
    train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
    train_y = np.array([[-2], [-3], [-7], [-7]])

    def reader():
        for i in xrange(train_y.shape[0]):
            yield train_x[i], train_y[i]

    return reader

feeding = {'x': 0, 'y': 1}

trainer.train(
    reader=paddle.batch(
        train_reader(), batch_size=1),
    feeding=feeding,
    event_handler=event_handler,
    num_passes=100)
