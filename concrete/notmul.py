from TrustOpinion import TrustOpinion
from matplotlib import pyplot as plt 
n = 10
a = TrustOpinion(1,0,0)
b = TrustOpinion(0,0,1)
a = a.binMult(b)
# print(a)
# res = []
# res.append(a)
# print(a)
# for i in range(n):
#     a = a.binMult(a)
#     print(a)
#     res.append(a)
# plt.plot(range(n+1), [i.t for i in res], label = "belief")
# plt.plot(range(n+1), [i.d for i in res],label = "disbelief")
# plt.plot(range(n+1), [i.u for i in res],label = "uncertainty")
# plt.plot(range(n+1), [i.a for i in res],label = "base rate")
# plt.legend()
# plt.show()