import tensorflow as tf

def build_vgg16(input_width, input_height, input_channels, n_classes):
    
    #placeholders
    X = tf.placeholder(tf.float32, shape=(None, input_width, 
    						input_height, input_channels))
    Y = tf.placeholder(tf.int64, shape=(None))

    lr_placeholder = tf.placeholder(tf.float32)
    
    #xavier initialization
    initializer = tf.contrib.layers.xavier_initializer(seed = 0)
    
 	  #conv2d_1_1 (224x244x3) -> (224x224x64)
    conv2d_1_1 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
    						kernel_initializer=initializer)
    
    #conv2d_1_2 (224x224x64) -> (224x224x64)
    conv2d_1_2 = tf.layers.conv2d(inputs=conv2d_1_1, filters=64, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #maxpool_1 (224x224x64) -> (128x128x64) 
    maxpool_1 = tf.layers.max_pooling2d(inputs=conv2d_1_2, pool_size=[2, 2],
        					strides=2, padding = 'SAME')

    #conv2d_2_1 (128x128x64) -> (128x128x128) 
    conv2d_2_1 = tf.layers.conv2d(inputs=maxpool_1, filters=128, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)
    
    #conv2d_2_2 (128x128x128) -> (128x128x128) 
    conv2d_2_2 = tf.layers.conv2d(inputs=conv2d_2_1, filters=128, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #maxpool_2 (128x128x128) -> (56x56x128)
    maxpool_2 = tf.layers.max_pooling2d(inputs=conv2d_2_2, pool_size=[2, 2],
        					strides=2, padding = 'SAME')

    #conv2d_3_1 (56x56x128) -> (56x56x256) 
    conv2d_3_1 = tf.layers.conv2d(inputs=maxpool_2, filters=256, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_3_2 (56x56x256) -> (56x56x256) 
    conv2d_3_2 = tf.layers.conv2d(inputs=conv2d_3_1, filters=256, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_3_3 (56x56x256) -> (56x56x256) 
    conv2d_3_3 = tf.layers.conv2d(inputs=conv2d_3_2, filters=256, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #maxpool_3 (56x56x256) -> (28x28x256)
    maxpool_3 = tf.layers.max_pooling2d(inputs=conv2d_3_3, pool_size=[2, 2],
        					strides=2, padding = 'SAME')

    #conv2d_4_1 (28x28x256) -> (28x28x512) 
    conv2d_4_1 = tf.layers.conv2d(inputs=maxpool_3, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_4_2 (28x28x512) -> (28x28x512) 
    conv2d_4_2 = tf.layers.conv2d(inputs=conv2d_4_1, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_4_3 (28x28x512) -> (28x28x512) 
    conv2d_4_3 = tf.layers.conv2d(inputs=conv2d_4_2, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_4_4 (28x28x512) -> (28x28x512) 
    conv2d_4_4 = tf.layers.conv2d(inputs=conv2d_4_3, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #maxpool_4 (28x28x512) -> (14x14x512)
    maxpool_4 = tf.layers.max_pooling2d(inputs=conv2d_4_4, pool_size=[2, 2],
        					strides=2, padding = 'SAME')


    #conv2d_5_1 (14x14x512) -> (14x14x512) 
    conv2d_5_1 = tf.layers.conv2d(inputs=maxpool_4, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #conv2d_5_2 (14x14x512) -> (14x14x512) 
    conv2d_5_2 = tf.layers.conv2d(inputs=conv2d_5_1, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

 	  #conv2d_5_3 (14x14x512) -> (14x14x512) 
    conv2d_5_3 = tf.layers.conv2d(inputs=conv2d_5_2, filters=512, kernel_size=[3,3], 
    strides=1, activation=tf.nn.relu, padding = 'SAME', 
        					kernel_initializer=initializer)

    #maxpool_5 (14x14x512) -> (7x7x512)
    maxpool_5 = tf.layers.max_pooling2d(inputs=conv2d_5_3, pool_size=[2, 2], 
    						strides=2, padding = 'SAME')
                            
    #flatten
    flatten = tf.contrib.layers.flatten(maxpool_5)

    #fc1 4096
    fc1 = tf.contrib.layers.fully_connected(flatten, num_outputs=4096, 
    							activation_fn=tf.nn.relu)

    #fc2 4096
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=4096, 
    							activation_fn=tf.nn.relu)    
    
    #output fc
    out = tf.contrib.layers.fully_connected(fc2, num_outputs=n_classes, 
    							activation_fn=None)   
    
    #conver to One-Hot Label
    one_hot = tf.one_hot(Y, depth=n_classes)
    
    #funco de perda/custo/erro
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=out) 
    
    #optimzer
    opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder).minimize(loss)
    
    #softmax
    softmax = tf.nn.softmax(out)
    
    #class
    class_ = tf.argmax(softmax,1)
    
    #acc
    compare_prediction = tf.equal(class_, Y)
    accuracy = tf.reduce_mean(tf.cast(compare_prediction, tf.float32))
    
    return X, Y, lr_placeholder, loss, opt, softmax, class_, accuracy
