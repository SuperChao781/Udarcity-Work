# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 13:01:24 2018

@author: wangyuchao
"""

from decimal import Decimal, getcontext

from vector_example import Vector

getcontext().prec = 10


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'
    LINE_IS_PARALLEL = 'Line is parallel'
    
    def __init__(self, normal_vector=None, constant_term=None):#normal_vector：法向量 constant_term：参数
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = Vector(normal_vector);#法向量的向量形式存储

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Line.first_nonzero_index(n)
            initial_coefficient = n[initial_index]

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector

        try:
            initial_index = Line.first_nonzero_index(n)
            terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)

#    以下代码均由本人编写
    #返回法向量的坐标值
    def __getitem__(self, item):
        return self.normal_vector.coordinates[item];


    def location_judge(self, l):#判断两条直线的位置关系 返回值：0：平行 1：重合 2：相交
        if(self.normal_vector.is_parallel_to(l.normal_vector)):#判断法向量是否平行
#           判断两个向量是否为同一向量 
            twopoint_vector = self.basepoint.minus(l.basepoint);#取两个直线上的基准向量的差
            if(twopoint_vector.is_zero() or (twopoint_vector.is_orthogonal_to(self.normal_vector)) ):#判断该向量是否与当前法向量正交
                return 1;#两向量重合
            else:
                return 0;#两向量平行
        else:
            return 2;#两向量相交        

    def is_cross_to(self,l):#判断是否相交
        #获取2个法向量
        vector1 = self.normal_vector;
        vector2 = l.normal_vector;
        return not (vector1.is_parallel_to(vector2));
        

    def calc_cross(self,l):#在2个向量既不平行也不重合的前提下，计算向量交点
        try:
            result = self.is_cross_to(l);
            if(result):
                #获取2个法向量的坐标
                a1 = self[0];
                b1 = self[1];
                a2 = l[0];
                b2 = l[1];
                #获取2个常参数
                k1 = self.constant_term;
                k2 = l.constant_term;
                x = (b2*k1-b1*k2)/(a1*b2-a2*b1);
                y = (a1*k2-a2*k1)/(a1*b2-a2*b1);
                return Vector((x,y));
            else:
                raise Exception(self.LINE_IS_PARALLEL)
        except Exception as e:
            if str(e) == self.LINE_IS_PARALLEL:
                raise Exception(self.LINE_IS_PARALLEL)
            else:
                raise e
            
        

class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps
        


def line_location(line1,line2):
    #判断两直线位置关系，如果相交，输出交点
    location = line1.location_judge(line2);
    print('直线位置关系判断:')
    print('直线1:'+str(line1));
    print('直线2:'+str(line2));
    if(location == 0):
        print('直线位置关系判断结果:平行');
    elif(location == 1):
        print('直线位置关系判断结果:重合');
    else:
        cross = line1.calc_cross(line2);
        print('直线位置关系判断结果:相交，交点为:' + '(' + str(cross[0]) + ',' + str(cross[1]) + ')');
    


def main():
#********************************************
#判断两直线位置关系，如果相交，输出交点
    myline1 = Line([4.046,2.836],1.21);
    myline2 = Line([10.115,7.09],3.025);
    line_location(myline1,myline2)
    
    myline1 = Line([7.204,3.182],8.68);
    myline2 = Line([8.172,4.114],9.883);
    line_location(myline1,myline2)

    myline1 = Line([1.182,5.562],6.744);
    myline2 = Line([1.773,8.343],9.525);
    line_location(myline1,myline2)
    
    

if __name__ == "__main__":
	main()        