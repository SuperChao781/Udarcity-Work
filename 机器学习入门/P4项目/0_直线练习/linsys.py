from decimal import Decimal, getcontext
from copy import deepcopy

from vector_example import Vector
from plain import Plain

getcontext().prec = 10


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'
    
    WRONG_INPUT = 'Wrong Input!Please check your input number'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)




    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plain.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret
        
        
#    以下代码均由本人编写
       
    def swap_rows(self, row1, row2):#交换等式
        pass # add your code here
        try:
            if(row1 == row2):
                return;
            elif( (row1 not in range(self.dimension)) and (row2 not in range(self.dimension)) ):
                raise Exception(self.WRONG_INPUT);
            else:
                exchange_plain = self.planes[row1];
                self.planes[row1]=self.planes[row2];
                self.planes[row2]=exchange_plain;
                
        except Exception as e: 
            if str(e) == self.WRONG_INPUT:
                print(self.WRONG_INPUT)
                return;
            else:
                raise e
    

    def multiply_coefficient_and_row(self, coefficient, row):#等式乘以常数
        pass # add your code here
        try:
            if(row not in range(self.dimension)):
                raise Exception(self.WRONG_INPUT);
            else:
                self.planes[row]=self.planes[row].times_scalar(coefficient);
                
        except Exception as e: 
            if str(e) == self.WRONG_INPUT:
                print(self.WRONG_INPUT)
                return;
            else:
                raise e


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):#将等式乘以常数，加到另1行上
        pass # add your code here
        try:
            if(row_to_add not in range(self.dimension)):
                raise Exception(self.WRONG_INPUT);
            else:
                added_plane = self.planes[row_to_add].times_scalar(coefficient);
                added_plane = added_plane.plus(self.planes[row_to_be_added_to]);
                self.planes[row_to_be_added_to] = added_plane;
                
        except Exception as e: 
            if str(e) == self.WRONG_INPUT:
                print(self.WRONG_INPUT)
                return;
            else:
                raise e
                
    def eliminate(self,row_start,col):#消去从指定行开始的指定未知数系数（起始行row_start除外）      
        if(MyDecimal(self[row_start][col]).is_near_zero()):
            return;#首行的指定系数为0，无法执行消元
        else:
            for i in range(row_start+1,len(self)):
                multiple = -1*(self[i][col]/self[row_start][col]);#计算被消元行系数相对于基准行系数的倍数的相反数
                self.add_multiple_times_row_to_row(multiple,row_start,i);#将基准行乘以该负倍率数，然后加到被消元行上
                
    def rowindex_of_first_nonzero_for_coefficient(self,row_start,col):#寻找指定行开始，第1个指定未知数系数不为0的行
        num_equation = len(self);
        for i in range(row_start,num_equation):
            if(not MyDecimal(self[i][col]).is_near_zero() ):
                return i;
        return -1;#从指定行开始，所有行指定未知数的系数均为0

    def compute_triangular_form(self):
        row_start = 0;#消元首行
        col = 0;#当前消元对象，0代表第1个未知数，1代表第2个未知数，依此类推
                
        #复制一个原始向量
        triangular_form = deepcopy(self);

        num_variable = triangular_form.dimension;
        num_equation = len(triangular_form);
        
        while(col < num_variable and row_start < num_equation):#如果已经消元到最后一行，或者所有系数都进行了一轮消元，则终止流程
            non_zero_row_index = triangular_form.rowindex_of_first_nonzero_for_coefficient(row_start,col);
            if(non_zero_row_index < 0):#没有必要消元
                 col += 1;
                 continue;
            elif(non_zero_row_index != row_start):#基准行的未知数系数是0，需要和后面的行交换
                triangular_form.swap_rows(row_start,non_zero_row_index);
            else:#基准行的未知数系数非0，可以作为基准
                pass
        
            triangular_form.eliminate(row_start,col);#以row_start行为基准，消去其后所有行的指定未知数系数
            row_start += 1;
            col += 1;                
        return triangular_form;



class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps

def LinearSystem_print(s,output_string):
    print('****************')
    print(output_string)
    print(s)
    print('****************')

    
    
def test1():#测试线性方程组的基本操作
    #********************************************
    #判断两直线位置关系，如果相交，输出交点
    
    p0 = Plain(normal_vector=Vector(['1','1','1']), constant_term='1')
    p1 = Plain(normal_vector=Vector(['0','1','0']), constant_term='2')
    p2 = Plain(normal_vector=Vector(['1','1','-1']), constant_term='3')
    p3 = Plain(normal_vector=Vector(['1','0','-2']), constant_term='2')
    
    s = LinearSystem([p0,p1,p2,p3])
    
    print(s.indices_of_first_nonzero_terms_in_each_row())
    print('{},{},{},{}'.format(s[0],s[1],s[2],s[3]))
    print(len(s))
    print(s)
    
    s[0] = p1
    print(s)
    
    print(MyDecimal('1e-9').is_near_zero())
    print(MyDecimal('1e-11').is_near_zero())
    
    p0 = Plain(normal_vector=Vector(['1','1','1']), constant_term='1')
    p1 = Plain(normal_vector=Vector(['0','1','0']), constant_term='2')
    p2 = Plain(normal_vector=Vector(['1','1','-1']), constant_term='3')
    p3 = Plain(normal_vector=Vector(['1','0','-2']), constant_term='2')
    
    s = LinearSystem([p0,p1,p2,p3])
    
    LinearSystem_print(s,'原始方程组')
    #交换等式
    s.swap_rows(0,1)
    print(s[2].normal_vector.is_parallel_to(p2.normal_vector))
    if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
        print('test case 1 failed')
    else:
        print('test case 1 successed')
    LinearSystem_print(s,'交换1-2行')
    
    s.swap_rows(1,3)
    if not (s[0] == p1 and s[1] == p3 and s[2] == p2 and s[3] == p0):
        print('test case 2 failed')
    else:
        print('test case 2 successed')
    LinearSystem_print(s,'交换2-4行')
    
    s.swap_rows(3,1)
    if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
        print('test case 3 failed')
    else:
        print('test case 3 successed')
    LinearSystem_print(s,'交换4-2行')
   
    s.multiply_coefficient_and_row(1,0)
    if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
        print('test case 4 failed')
    else:
        print('test case 4 successed')
    LinearSystem_print(s,'方程1乘1')

    s.multiply_coefficient_and_row(-1,2)
    if not (s[0] == p1 and
            s[1] == p0 and
            s[2] == Plain(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
            s[3] == p3):
        print('test case 5 failed')
    else:
        print('test case 5 successed')
    LinearSystem_print(s,'方程3乘-1')
    
    s.multiply_coefficient_and_row(10,1)
    if not (s[0] == p1 and
            s[1] == Plain(normal_vector=Vector(['10','10','10']), constant_term='10') and
            s[2] == Plain(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
            s[3] == p3):
        print('test case 6 failed')
    else:
        print('test case 6 successed')
    LinearSystem_print(s,'方程2乘10')
    
    s.add_multiple_times_row_to_row(0,0,1)
    if not (s[0] == p1 and
            s[1] == Plain(normal_vector=Vector(['10','10','10']), constant_term='10') and
            s[2] == Plain(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
            s[3] == p3):
        print('test case 7 failed')
    else:
        print('test case 7 successed')
    LinearSystem_print(s,'方程1乘0，加到方程2')
    
    s.add_multiple_times_row_to_row(1,0,1)
    if not (s[0] == p1 and
            s[1] == Plain(normal_vector=Vector(['10','11','10']), constant_term='12') and
            s[2] == Plain(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
            s[3] == p3):
        print('test case 8 failed')
    else:
        print('test case 8 successed')
    LinearSystem_print(s,'方程1乘1，加到方程2')
    
    s.add_multiple_times_row_to_row(-1,1,0)
    if not (s[0] == Plain(normal_vector=Vector(['-10','-10','-10']), constant_term='-10') and
            s[1] == Plain(normal_vector=Vector(['10','11','10']), constant_term='12') and
            s[2] == Plain(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
            s[3] == p3):
        print('test case 9 failed')
    else:
        print('test case 9 successed')
    LinearSystem_print(s,'方程2乘-1，加到方程1')
    
def test2():#线性方程组阶梯化
    
    p1 = Plain(normal_vector=Vector(['1','1','1']), constant_term='1')
    p2 = Plain(normal_vector=Vector(['0','1','1']), constant_term='2')
    s = LinearSystem([p1,p2])
    t = s.compute_triangular_form()
    print('*******************************')
    if not (t[0] == p1 and
            t[1] == p2):
        print('test case 1 failed')
    else:
        print('test case 1 successed')
        LinearSystem_print(s,'原始方程')
        LinearSystem_print(t,'阶梯化后方程')
    print('*******************************')
    
    p1 = Plain(normal_vector=Vector(['1','1','1']), constant_term='1')
    p2 = Plain(normal_vector=Vector(['1','1','1']), constant_term='2')
    s = LinearSystem([p1,p2])
    t = s.compute_triangular_form()
    print('*******************************')
    if not (t[0] == p1 and
            t[1] == Plain(constant_term='1')):
        print('test case 2 failed')
    else:
        print('test case 2 successed')
        LinearSystem_print(s,'原始方程')
        LinearSystem_print(t,'阶梯化后方程')
    print('*******************************')
       
    p1 = Plain(normal_vector=Vector(['1','1','1']), constant_term='1')
    p2 = Plain(normal_vector=Vector(['0','1','0']), constant_term='2')
    p3 = Plain(normal_vector=Vector(['1','1','-1']), constant_term='3')
    p4 = Plain(normal_vector=Vector(['1','0','-2']), constant_term='2')
    s = LinearSystem([p1,p2,p3,p4])
    t = s.compute_triangular_form()
    print('*******************************')
    if not (t[0] == p1 and
            t[1] == p2 and
            t[2] == Plain(normal_vector=Vector(['0','0','-2']), constant_term='2') and
            t[3] == Plain()):
        print('test case 3 failed')
    else:
        print('test case 3 successed')
        LinearSystem_print(s,'原始方程')
        LinearSystem_print(t,'阶梯化后方程')
    print('*******************************')
    
    p1 = Plain(normal_vector=Vector(['0','1','1']), constant_term='1')
    p2 = Plain(normal_vector=Vector(['1','-1','1']), constant_term='2')
    p3 = Plain(normal_vector=Vector(['1','2','-5']), constant_term='3')
    s = LinearSystem([p1,p2,p3])
    t = s.compute_triangular_form()
    print('*******************************')
    if not (t[0] == Plain(normal_vector=Vector(['1','-1','1']), constant_term='2') and
            t[1] == Plain(normal_vector=Vector(['0','1','1']), constant_term='1') and
            t[2] == Plain(normal_vector=Vector(['0','0','-9']), constant_term='-2')):
        print('test case 4 failed')
    else:
        print('test case 4 successed')
        LinearSystem_print(s,'原始方程')
        LinearSystem_print(t,'阶梯化后方程')
    print('*******************************')
    
    
    
    
    


def main():
    #test1();
    test2();

    
if __name__ == "__main__":
	main()        