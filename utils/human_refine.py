import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

def human_refine(y, l):

    y1 = y[-7:]
    l1 = l[:7]
    r1 = human_refine7(y1,l1)
    yl1 = np.append(y1, l1, axis=0)
    yr1 = np.append(y1, r1, axis=0)
    # plt.plot(yl1)
    # plt.plot(yr1)
    # plt.legend(labels = ['l','ml'])
    # plt.show()

    y2 = y[-7:]
    l2 = l[7:14]
    r2 = human_refine7(y2,l2)

    yl2 = np.append(yl1, l2, axis=0)
    yr2 = np.append(yr1, r2, axis=0)
    # plt.plot(yl2)
    # plt.plot(yr2)
    # plt.legend(labels = ['l','ml'])
    # plt.show()

    y3 = y[-13:-6]
    l3 = l[-7:]
    r3 = human_refine7(y3,l3)

    ret = np.append(r1, r2, axis=0)
    ret = np.append(ret, r3[-1:], axis=0)
    print(len(ret))
    # plt.plot(np.append(y,l, axis=0))
    # plt.plot(np.append(y,ret, axis=0))
    # plt.legend(labels = ['l','ml'])
    # plt.show()
    return ret 


def human_refine7(y,l):
    """
        y: 
        l: 
    """ 
    # lx = list()
    # sx = x.split()
    # for i in sx:
    #     lx.append(float(i))
    # lx = np.array(lx)
    # y = lx[:7]
    # l = lx[7:]
    base_y = np.min(y)
    top_y = np.max(y)
    topy_i = np.argmax(y)
    basey_i = np.argmin(y)
    base_l = l[basey_i]
    
    align_l = l + (base_y - base_l)
    # base_align_l = np.min(align_l)

    top_l = align_l[topy_i]
    ampl_y = top_y - base_y 
    ampl_l = top_l - base_y
    ml =  (align_l-base_y)* (ampl_y / ampl_l) + base_y - (base_y - base_l)#align_l

    # plt.plot(np.append(y,l, axis=0))
    # plt.plot(np.append(y,ml, axis=0))
    # plt.legend(labels = ['l','ml'])

    # plt.plot(y, label='y')
    # plt.plot(l, label='l')
    # plt.plot(align_l)
    # plt.plot(base_y-l+l)
    # plt.plot(ml, label='ml')
    # plt.legend(labels = ['y', 'l','bl', 'base','ml'])
    # plt.show()
    return ml

if __name__ == '__main__':
    x = '5.54664078 5.31953125 5.30931992 5.31677745 5.31088775 5.30283559 5.39570667 5.49412503 5.28643101 5.19794108 5.25091645 5.25703253 5.21840715 5.35296365 5.35271212 5.27627978 5.23612961 5.23217262 5.23220137 5.25257955 5.33653738 5.35901359 5.2994004  5.27004793 5.27207461 5.27569823 5.28705566 5.34016516 5.35305509'
    lx = list()
    sx = x.split()
    print(len(sx))

    for i in sx:
        lx.append(float(i))
    print(len(lx))
    base = np.array(lx[:14])
    pred = np.array(lx[14:])
    refine1 = human_refine(base, pred)
    # print(len(refine1))
    print(len(pred))
    plt.plot(np.append(base,pred, axis=0))
    plt.plot(np.append(base,refine1, axis=0))
    plt.legend(labels = ['l','ml'])
    plt.show()
    # refine2 = human_refine(base, pred[7:14])