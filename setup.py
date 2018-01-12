from distutils.core import setup
setup(
  name = 'sparsemf',
  packages = ['sparsemf'], 
  version = '0.11',
  description = 'A matrix factorization recommender which runs on top of NumPy and SciPy. Developed with a focus on speed, and highly sparse matrices.',
  author = 'Jake Hawkesworth',
  author_email = 'jeh0753@gmail.com',
  url = 'https://github.com/jeh0753/sparseMF', 
  download_url = 'https://github.com/jeh0753/sparseMF/archive/0.11.tar.gz', 
  keywords = ["imputation","matrix factorization","recommender","recommendation engine","collaborative filtering","softimpute"], 
  classifiers = [],
)
