import time
import numpy as np
from keras.preprocessing.image import save_img
from keras import layers
from keras import backend as K
from keras.models import load_model

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def process_image(x, former):
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()

def visualize_layer(model, layer_name, step=1., epochs=15, upscaling_steps=9, upscaling_factor=1.2, output_dim=(48, 48), filter_range=(0, None)):

    def _generate_filter_image(input_img, layer_output, filter_index):

        s_time = time.time()

        loss = K.mean(layer_output[:, :, :, filter_index])

        grads = K.gradients(loss, input_img)[0]

        grads = normalize(grads)

        iterate = K.function([input_img], [loss, grads])

        input_img_data = np.random.random((1, 48, 48, 1))

        for up in reversed(range(upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # some filters get stuck to 0, we can skip them
                if loss_value <= K.epsilon():
                    return None

            # Upscale
            img = deprocess_image(input_img_data[0])
            input_img_data = [process_image(img, input_img_data[0])]

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index, loss_value, e_time - s_time))
        return img, loss_value

    def _draw_filters(filters):
        
        n = int(np.floor(np.sqrt(len(filters))))

        filters.sort(key=lambda x: x[1], reverse=True)
        filters = filters[:n * n]

        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 1), dtype='uint8')

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, _ = filters[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[ width_margin: width_margin + output_dim[0], height_margin: height_margin + output_dim[1], :] = img

        # save the result to disk
        save_img('fig2_1.jpg', stitched_filters)

    # this is the placeholder for the input images
    assert len(model.inputs) == 1
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # Compute to be processed filter range
    filter_lower = filter_range[0]
    filter_upper = (filter_range[1] if filter_range[1] is not None else len(output_layer.get_weights()[1]))
    assert(filter_lower >= 0 and filter_upper <= len(output_layer.get_weights()[1]) and filter_upper > filter_lower)
    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

    # iterate through each filter and generate its corresponding image
    processed_filters = []
    for f in range(filter_lower, filter_upper):

        img_loss = _generate_filter_image(input_img, output_layer.output, f)

        if img_loss is not None:
            processed_filters.append(img_loss)

    print('{} filter processed.'.format(len(processed_filters)))
    # Finally draw and store the best filters to disk
    _draw_filters(processed_filters)

if __name__ == '__main__':
    ##### layer to draw
    LAYER_NAME = 'conv2d_1'
    
    ##### load model
    model = load_model('model1.h5')

    ##### visaulize the image that activate the filter
    visualize_layer(model, LAYER_NAME)