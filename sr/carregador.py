from functools import reduce
from .utils import *
from tensorflow.python.data.experimental import AUTOTUNE
import nibabel as nib

size = 50

class Loader:
    def __init__(self,
                 size=50,
                 subset='train',
                 images_dir='OS/JPEGImages',
                 caches_dir='OS/caches',
                 percent=0.5):

        _files = sorted(os.listdir(images_dir)) # [:size]


        idxs = int(size*percent) 
        if subset == 'train':
            self.input_ids = [file for file in _files if "volume" in file ][:idxs]
            self.target_ids = [file for file in _files if "label" in file ][:idxs]
        elif subset == 'valid':
            self.input_ids = [file for file in _files if "volume" in file ][idxs:size]
            self.target_ids = [file for file in _files if "label" in file ][idxs:size]
        else:
            raise ValueError("subset must be 'train' or 'valid'")
            
        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir
        # self.caches_dir = f'{caches_dir}/{downgrade}' 

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)
    @tf.autograph.experimental.do_not_convert
    def load(self, random_transform = False):
        # .unbatch
        # print("entrou")
        entrada = self.input_dataset()
        # print(entrada)
        # entrada = tf.data.Dataset.list_files(f'{self.images_dir}/volume*')
        entrada
        # print(list(entrada))
        # entrada = entrada.map(lambda filename: load_lazy(filename))
        entrada = tf.map_fn(fn = lambda filename: load_lazy(filename), 
                elems=entrada, fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, None], dtype =tf.float64))
        print(entrada.shape)  
        entrada = tf.data.Dataset.from_tensor_slices(entrada)
        entrada = entrada.reduce(None, lambda lista, x: tf.concat(lista, x, axis=2) if lista is not None else x)
        print(entrada)  
        # entrada = tf.reduce(fn = lambda filename: load_lazy(filename), 
        #         elems=entrada, fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, None], dtype =tf.float64))
        # , num_parallel_calls=AUTOTUNE
        
        if random_transform:
            ds = ds.map(lambda entrada, target: random_crop(entrada, target), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
            ds = ds.batch()
        return entrada

    def load_l(self, random_transform = True):
        # .unbatch
        # print("entrou")
        # print(entrada)
        entrada = self.input_dataset()
        targets = self.target_dataset()

        ds = tf.data.Dataset.zip((entrada, targets))

        if random_transform:
            # ds = ds.map(lambda entrada, target: random_crop(entrada, target), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=2), tf.expand_dims(y, axis=2)) , num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

        
        # reduce(lambda lista, element: np.concatenate(lista, element) if lista is not None else element, [[1,2,3,4,5], [11,22,33,34,35, 6]])
        return ds
    
    def one_hot(self, x):
        indices = [1, 2, 3, 4, 5, 6]
        x = tf.cast(x, dtype=tf.uint8)
        x = tf.one_hot(x, len(indices))
        return x

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        # ds = tf.data.Dataset.zip((self.input_dataset(), self.target_dataset()))
        
        entrada = self.input_dataset()
        targets = self.target_dataset()

        ds = tf.data.Dataset.zip((entrada, targets))

        if random_transform:
            ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=2), one_hot(y)) , num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda entrada, target: random_crop(entrada, target), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        

        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def target_dataset(self):
        if not os.path.exists(self._target_images_dir()):
            raise ValueError("subset not found ")
            
        ds = self._images_dataset(self._target_image_files()).cache(self._target_cache_file())
        
        if not os.path.exists(self._target_cache_index()):
            self._populate_cache(ds, self._target_cache_file())
        
        return ds

        # return self._images_dataset(self._target_image_files())

    def input_dataset(self):
        if not os.path.exists(self._input_images_dir()):
            raise ValueError("subset not found ")
            
        ds = self._images_dataset(self._input_image_files()).cache(self._input_cache_file())
        
        if not os.path.exists(self._input_cache_index()):
            self._populate_cache(ds, self._input_cache_file())
        
        return ds
        # return self._images_dataset(self._input_image_files())

    def _target_image_files(self):
        images_dir = self._target_images_dir()
        return [os.path.join(images_dir, f'{image_id}') for image_id in self.target_ids]

    def _input_image_files(self):
        images_dir = self._input_images_dir()
        return [os.path.join(images_dir, f'{image_id}') for image_id in self.input_ids]

    def _input_image_file(self, image_id):
        return f'{image_id}.png'

    def _target_images_dir(self):
        return os.path.join(self.images_dir)

    def _input_images_dir(self):
        return os.path.join(self.images_dir)

    def _target_cache_file(self):
        return os.path.join(self.caches_dir, f'seg_{self.subset}_target.cache')

    def _input_cache_file(self):
        return os.path.join(self.caches_dir, f'seg_{self.subset}_input.cache')

    def _target_cache_index(self):
        return f'{self._target_cache_file()}.index'

    def _input_cache_index(self):
        return f'{self._input_cache_file()}.index'
    def reduce_element(lista, element):
        if lista is not None:
            if element is not None:
                return np.dstack((lista, element))
            else:
                lista 
        else:
            element 
    @staticmethod
    def _images_dataset(image_files):  
        images = [load_lazy(x) for x in image_files]
        images = reduce(lambda lista, element: np.vstack((lista, element)) if lista is not None else element, images)
        return tf.data.Dataset.from_tensor_slices(images)   
        # return tf.constant(image_files)   

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')



# data_loader_train = Loader(scale=4, 
#                 subset='train', 
#                 downgrade='bicubic', 
#                 images_dir="dataset/S2TLD/JPEGImages",
#                 caches_dir="dataset/S2TLD/caches",
#                 percent=0.7)

# print("="*100)
# train = data_loader_train.dataset(4, random_transform=True)
# print(train)
# print(len(os.listdir("dataset/S2TLD/JPEGImages")))
# load_image("dataset/S2TLD/JPEGImages/2020-03-30 11_30_03.690871079.jpg")
def __main__():
    pass
if __name__ == "__main__":
    pass