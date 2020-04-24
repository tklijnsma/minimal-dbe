def construct_model(
        quantized,
        saved_model_dir = None,
        starting_weights_directory = None,
        is_frozen = False,
        is_training = True,
        size = 64,
        weights_file = None
        ):
    from azureml.accel.models import Resnet50, QuantizedResnet50
    import tensorflow as tf
    from keras import backend as K
    # Convert images to 3D tensors [width,height,channel]
    in_images, image_tensors = preprocess_images(size=size)
    # Construct featurizer using quantized or unquantized ResNet50 model
    if not quantized:
        # featurizer = Resnet50(saved_model_dir, is_frozen=is_frozen, custom_weights_directory = starting_weights_directory)
        featurizer = Resnet50(saved_model_dir)
    else:
        featurizer = QuantizedResnet50(saved_model_dir, is_frozen=is_frozen, custom_weights_directory = starting_weights_directory)
    
    features = featurizer.import_graph_def(input_tensor=image_tensors, is_training=is_training)
    # Construct classifier
    with tf.name_scope('classifier'):
        classifier = construct_classifier()
        preds = classifier(features)
    # Initialize weights
    sess = tf.get_default_session()
    tf.global_variables_initializer().run()
    if not is_frozen:
        print('Restoring weights from featurizer into session')
        featurizer.restore_weights(sess)
    if not(weights_file is None):
        print("loading classifier weights from", weights_file)
        classifier.load_weights(weights_file)
    elif starting_weights_directory is not None:
        print("loading classifier weights from", starting_weights_directory+'/class_weights_best.h5')
        classifier.load_weights(starting_weights_directory+'/class_weights_best.h5')
    return in_images, image_tensors, features, preds, featurizer, classifier 


def normalize_and_rgb(images): 
    import numpy as np
    #normalize image to 0-255 per image.
    image_sum = 1/np.sum(np.sum(images,axis=1),axis=-1)
    given_axis = 0
    # Create an array which would be used to reshape 1D array, b to have 
    # singleton dimensions except for the given axis where we would put -1 
    # signifying to use the entire length of elements along that axis  
    dim_array = np.ones((1,images.ndim),int).ravel()
    dim_array[given_axis] = -1
    # Reshape b with dim_array and perform elementwise multiplication with 
    # broadcasting along the singleton dimensions for the final output
    image_sum_reshaped = image_sum.reshape(dim_array)
    images = images*image_sum_reshaped*255
    # make it rgb by duplicating 3 channels.
    images = np.stack([images, images, images],axis=-1)
    return images


def image_with_label(train_file, istart,iend):
    import tables
    import numpy as np
    f = tables.open_file(train_file, 'r')
    a = np.array(f.root.img_pt)[istart:iend].copy() # Images
    b = np.array(f.root.label)[istart:iend].copy() # Labels
    f.close()
    return normalize_and_rgb(a),b


def count_events(train_files):
    import tables
    n_events = 0
    for train_file in train_files:
        f = tables.open_file(train_file, 'r')
        n_events += f.root.label.shape[0]
        f.close()
    return n_events


def preprocess_images(size=64):
    import tensorflow as tf
    # Create a placeholder for our incoming images
    in_height = size
    in_width = size
    in_images = tf.placeholder(tf.float32)
    in_images.set_shape([None, in_height, in_width, 3])
    # Resize those images to fit our featurizer
    if size==64:
        out_width = 224
        out_height = 224
        image_tensors = tf.image.resize_images(in_images, [out_height,out_width])
        image_tensors = tf.to_float(image_tensors)
    elif size==224:
        image_tensors = in_images
    return in_images, image_tensors


def construct_classifier():
    from keras.layers import Dropout, Dense, Flatten, Input
    from keras.models import Model
    from keras import backend as K
    import tensorflow as tf
    K.set_session(tf.get_default_session())
    FC_SIZE = 1024
    NUM_CLASSES = 2
    in_layer = Input(shape=(1, 1, 2048,),name='input_1')
    x = Dense(FC_SIZE, activation='relu', input_dim=(1, 1, 2048,),name='dense_1')(in_layer)
    x = Flatten(name='flatten_1')(x)
    preds = Dense(NUM_CLASSES, activation='softmax', input_dim=FC_SIZE, name='classifier_output')(x)
    model = Model(inputs = in_layer, outputs = preds)
    return model


def check_model(preds, features, in_images, train_files, classifier):
    import tensorflow as tf
    from keras import backend as K
    sess = tf.get_default_session()
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))
    a, b = image_with_label(train_files[0],0,1)
    c = classifier.layers[-1].weights[0]
    d = classifier.layers[-1].weights[1]
    print(" image:    ", a)
    print(" label:    ", b)
    print(" features: ", sess.run(features, feed_dict={in_images: a,
                                   in_labels: b,
                                   K.learning_phase(): 0}))
    print(" weights:  ", sess.run(c))
    print(" biases:   ", sess.run(d))    
    print(" preds:    ", sess.run(preds, feed_dict={in_images: a,
                                   in_labels: b,
                                   K.learning_phase(): 0}))


def chunks(files, chunksize, max_q_size=4, shuffle=True): 
    """Yield successive n-sized chunks from a and b.""" 
    import tables
    import numpy as np
    for train_file in files: 
        f = tables.open_file(train_file, 'r') 
        nrows = f.root.label.nrows
        for istart in range(0,nrows,max_q_size*chunksize):  
            a = np.array(f.root.img_pt[istart:istart+max_q_size*chunksize]) # Images 
            b = np.array(f.root.label[istart:istart+max_q_size*chunksize]) # Labels 
            if shuffle: 
                c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)] # shuffle within queue size
                np.random.shuffle(c)
                test_images = c[:, :a.size//len(a)].reshape(a.shape)
                test_labels = c[:, a.size//len(a):].reshape(b.shape)
            else:
                test_images = a
                test_labels = b
            for jstart in range(0,len(test_labels),chunksize): 
                yield normalize_and_rgb(test_images[jstart:jstart+chunksize].copy()),test_labels[jstart:jstart+chunksize].copy(), len(test_labels[jstart:jstart+chunksize].copy())  
        f.close()


def test_model(preds, in_images, test_files, chunk_size=64, shuffle=True):
    """Test the model"""
    import tensorflow as tf
    from keras import backend as K
    from keras.objectives import binary_crossentropy 
    import numpy as np
    from keras.metrics import categorical_accuracy
    from tqdm import tqdm
    
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))
    
    cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
    accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
    auc = tf.metrics.auc(tf.cast(in_labels, tf.bool), preds)
   
    n_test_events = count_events(test_files)
    chunk_num = int(n_test_events/chunk_size)+1
    preds_all = []
    label_all = []
    
    sess = tf.get_default_session()
    sess.run(tf.local_variables_initializer())
    
    avg_accuracy = 0
    avg_auc = 0
    avg_test_loss = 0
    is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(test_files, chunk_size, shuffle=shuffle),total=chunk_num):
        test_loss, accuracy_result, auc_result, preds_result = sess.run([cross_entropy, accuracy, auc, preds],
                        feed_dict={in_images: img_chunk,
                                   in_labels: label_chunk,
                                   K.learning_phase(): 0,
                                   is_training: False})
        avg_test_loss += test_loss * real_chunk_size / n_test_events
        avg_accuracy += accuracy_result * real_chunk_size / n_test_events
        avg_auc += auc_result[0]  * real_chunk_size / n_test_events 
        preds_all.extend(preds_result)
        label_all.extend(label_chunk)
    
    print("test_loss = ", "{:.3f}".format(avg_test_loss))
    print("Test Accuracy:", "{:.3f}".format(avg_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_auc))
    
    return avg_test_loss, avg_accuracy, avg_auc, np.asarray(preds_all).reshape(n_test_events,2), np.asarray(label_all).reshape(n_test_events,2)
