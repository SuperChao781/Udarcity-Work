# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:24:26 2018

@author: wangyuchao
"""

from decimal import Decimal, getcontext

from vector_example import Vector

getcontext().prec = 30


class Plain(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 3

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = Vector(normal_vector)

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Plain.first_nonzero_index(n)
            initial_coefficient = n[initial_index]

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Plain.NO_NONZERO_ELTS_FOUND_MSG:
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
            initial_index = Plain.first_nonzero_index(n)
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
        raise Exception(Plain.NO_NONZERO_ELTS_FOUND_MSG)


#    以下代码均由本人编写
    #返回法向量的坐标值
    def __getitem__(self, item):
        return self.normal_vector.coordinates[item];


    #判断两平面是否重合
    def __eq__(self,l):
        if(self.normal_vector.is_parallel_to(l.normal_vector)):
            minus_vector = self.basepoint.minus(l.basepoint);
            if(minus_vector.is_zero() or minus_vector.is_orthogonal_to(l.normal_vector)):#判断该向量是否与当前法向量正交
                return True;
            else:
                return False;
            
        else:
            return False;
    
    #判断两平面是否平行
    def is_parallel_to(self,l):
        return (self.normal_vector.is_parallel_to(l.normal_vector));


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps
        
        
def plain_location(plain1,plain2):
    #判断两直线位置关系，如果相交，输出交点
    print('平面位置关系判断:')
    print('平面1:'+str(plain1));
    print('平面2:'+str(plain2));
    if(plain1==plain2):
        print('平面位置关系判断结果:重合');
    elif(plain1.is_parallel_to(plain2)):
        print('平面位置关系判断结果:平行');
    else:
        print('平面位置关系判断结果:相交');
        #cross = line1.calc_cross(line2);
        #print('平面位置关系判断结果:相交，交点为:' + '(' + str(cross[0]) + ',' + str(cross[1]) + ')');
    
    
    
def main():
#********************************************
#判断两直线位置关系，如果相交，输出交点
    myplain1 = Plain([-0.412,3.806,0.728],-3.46);
    myplain2 = Plain([1.03,-9.515,-1.822],8.65);
    plain_location(myplain1,myplain2)
    
    myplain1 = Plain([2.611,5.528,0.283],4.6);
    myplain2 = Plain([7.715,8.306,5.342],3.76);
    plain_location(myplain1,myplain2)

    myplain1 = Plain([-7.926,8.625,-7.217],-7.952);
    myplain2 = Plain([-2.692,2.875,-2.404],-2.443);
    plain_location(myplain1,myplain2)
    
    

if __name__ == "__main__":
	main()             
        
        
        
        