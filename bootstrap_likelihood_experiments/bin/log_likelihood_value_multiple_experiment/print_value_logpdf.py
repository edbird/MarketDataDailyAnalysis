
import math
import scipy

print(scipy.stats.norm.logpdf(0, 0, 1))
print(scipy.stats.norm.pdf(0, 0, 1))
print(math.log(scipy.stats.norm.pdf(0, 0, 1)))
print( (1.0 / 2.0) / math.sqrt(math.pi) )