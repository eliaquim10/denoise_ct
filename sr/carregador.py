from functools import reduce
from .utils import *
from tensorflow.python.data.experimental import AUTOTUNE
import nibabel as nib

size = 50

class Loader:
    def __init__(self,
                 size=50,
                 repeat = 1,
                 batch_size = 8,
                 load_type = "nii.gz",
                 subset='train',
                 images_dir='OS/JPEGImages',
                 caches_dir='OS/caches',
                 percent=0.5):
        """
            _files = sorted(os.listdir(images_dir)) # [:size]


            idxs = int(size*percent) 

            if(load_type == "nii.gz"):

                if subset == 'train':
                    self.input_ids = [file for file in _files if "volume" in file ][:idxs]
                    self.target_ids = [file for file in _files if "label" in file ][:idxs]
                elif subset == 'valid':
                    self.input_ids = [file for file in _files if "volume" in file ][idxs:size]
                    self.target_ids = [file for file in _files if "label" in file ][idxs:size]
                else:
                    raise ValueError("subset must be 'train' or 'valid'")
            else:
                if subset == 'train':
                    self.input_ids = [file for file in os.listdir(os.path.join(images_dir, "volume"))][:idxs]
                    self.target_ids = [file for file in os.listdir(os.path.join(images_dir, "label"))][:idxs]
                    print("size", (self.input_ids))
                elif subset == 'valid':
                    self.input_ids = [file for file in os.listdir(os.path.join(images_dir, "volume"))][idxs:size]
                    self.target_ids = [file for file in os.listdir(os.path.join(images_dir, "label"))][idxs:size]
                else:
                    raise ValueError("subset must be 'train' or 'valid'")
        """
            
        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir
        self.batch_size = batch_size
        self.repeat = repeat
        self.size = size
        self.percent = percent
        self.load_type = load_type
        # self.caches_dir = f'{caches_dir}/{downgrade}' 

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    # @tf.autograph.experimental.do_not_convert
    def load(self):
        # ds = tf.data.Dataset.zip((self.input_dataset(), self.target_dataset()))
        def split(s):
            print(" s ",s)
            s = s.split("/")[-1] 
            s = s.split("_") 
            print(" s ",s)
            return int(s[1]), int(s[3].split(".")[0])
        if(self.load_type != "nii.gz"):
            # length = 10
            entrada = tf.constant(self._input_image_files())
            # entrada = [split(i) for i in entrada]
            # entrada = [tf.constant(entrada[i:i + length]) for i in range(0, len(entrada), length)]
            # entrada = tf.cast(entrada, dtype=tf.int16)
            # print(entrada)
            # exit()

            targets = tf.constant(self._target_image_files())
            # targets = [split(i) for i in targets]
            # targets = [tf.constant(targets[i:i + length]) for i in range(0, len(targets), length)]
            # targets = tf.cast(targets, dtype=tf.strings)

            
            entrada = tf.data.Dataset.from_tensor_slices(entrada)
            targets = tf.data.Dataset.from_tensor_slices(targets)

            ds = tf.data.Dataset.zip((entrada, targets))
        else:
            entrada = tf.constant(self._input_image_files())
            targets = tf.constant(self._target_image_files())
            
            entrada = tf.data.Dataset.from_tensor_slices(entrada)
            targets = tf.data.Dataset.from_tensor_slices(targets)

            ds = tf.data.Dataset.zip((entrada, targets))

        return ds

    def load_file(self, entrada, target, batch_size=16, repeat_count=None, random_transform=True):
        def squeze(x):
            print("squeze", x.shape)
            # tf.squeeze(
            return one_hot(x)

        if(self.load_type != "nii.gz"):
            entrada, target = self._images_load_npy(entrada, target)        
        else:
            entrada, target = self._images_load(entrada, target)
        # entrada, target =  bytes.decode(entrada.numpy()), bytes.decode(target.numpy())
        # print(entrada, target)
        
        ds = tf.data.Dataset.zip((entrada, target))
        # if(self.load_type != "nii.gz"):
        #     ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=2), squeze(y)) , num_parallel_calls=AUTOTUNE)
        #     # ds = ds.map(lambda x, y: (x, one_hot(y)) , num_parallel_calls=AUTOTUNE)
        # else:
        ds = ds.map(lambda x, y: (tf.expand_dims(x, axis=2), one_hot(y)) , num_parallel_calls=AUTOTUNE)

        if random_transform:
            ds = ds.map(lambda entrada, target: random_crop(entrada, target), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
            ds = ds.map(clip, num_parallel_calls=AUTOTUNE)
        

        ds = ds.batch(self.batch_size)
        ds = ds.repeat(self.repeat)
        # ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    def get_elements(self):
        dataset = self.load()
        for file_entrada, file_target in dataset:
            dataset = self.load_file(entrada = file_entrada, target = file_target)
            for entrada, target in dataset:
                yield entrada, target

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
            ds = ds.map(clip, num_parallel_calls=AUTOTUNE)
        

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


        idxs = int(self.size*self.percent) 
        if self.load_type == "nii.gz":
            _files = [element for element in sorted(os.listdir(images_dir)) if "label" in element ] # [:size]
        else:
            _files = sorted(os.listdir(images_dir)) # [:size]


        if self.subset == 'train':
            target_ids = _files[:idxs]
        elif self.subset == 'valid':
            target_ids = _files[idxs:size]
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        return [os.path.join(images_dir, f'{image_id}') for image_id in target_ids]

    def _input_image_files(self):
        images_dir = self._input_images_dir()


        idxs = int(self.size*self.percent) 
        if self.load_type == "nii.gz":
            _files = [element for element in sorted(os.listdir(images_dir)) if "volume" in element ] # [:size]
        else:
            _files = sorted(os.listdir(images_dir)) # [:size]

        if self.subset == 'train':
            input_ids = _files[:idxs]
        elif self.subset == 'valid':
            input_ids = _files[idxs:size]
        else:
            raise ValueError("subset must be 'train' or 'valid'")
        return [os.path.join(images_dir, f'{image_id}') for image_id in input_ids]
    
    def one_hot(self, x):
        indices = [1, 2, 3, 4, 5, 6]
        x = tf.cast(x, dtype=tf.uint8)
        x = tf.one_hot(x, len(indices))
        return x

    def _input_image_file(self, image_id):
        return f'{image_id}.png'

    def _target_images_dir(self):
        if(self.load_type != "nii.gz"):
            return os.path.join(self.images_dir, "label")
        return os.path.join(self.images_dir)

    def _input_images_dir(self):
        if(self.load_type != "nii.gz"):
            return os.path.join(self.images_dir, "volume")
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

    def to_str(self, entrada, target):  
        return bytes.decode(entrada.numpy()), bytes.decode(target.numpy())

    @staticmethod
    def _images_dataset(image_files):  
        images = [load_lazy(x) for x in image_files]
        images = reduce(lambda lista, element: np.vstack((lista, element)) if lista is not None else element, images)
        images = tf.cast(images, dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices(images)   
        # return tf.constant(image_files)   


    @staticmethod
    def _images_load(entrada, target):  
        entrada, target =  bytes.decode(entrada.numpy()), bytes.decode(target.numpy())
        entrada, target = load_lazy(entrada), load_lazy(target)
        return tf.data.Dataset.from_tensor_slices(entrada), tf.data.Dataset.from_tensor_slices(target) 

    @staticmethod
    def _images_load_npy(entrada, target):     
        entrada, target =  bytes.decode(entrada.numpy()), bytes.decode(target.numpy())
        entrada, target = np.load(entrada), np.load(target)
        entrada, target = entrada.transpose(2,0,1), target.transpose(2,0,1)
        
        return tf.data.Dataset.from_tensor_slices(entrada), tf.data.Dataset.from_tensor_slices(target) 

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