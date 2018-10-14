import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I
        
def dot_product(vector_one, vector_two):
    sum = 0;
    if(len(vector_one)!=len(vector_two)):
        raise(ValueError, "The dimesion is not equal!");
    
    for m,n in zip(vector_one,vector_two):
        sum = sum + m*n;
    
    return sum;

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)#行数
        self.w = len(grid[0])#列数

    #############学员自定义函数 begin################
    '''
    #通用函数编程
    '''
    def get_row(self, row):
        if row>= self.h:
            raise(ValueError, "Input is too large.");
        return self.g[row];
    
    def get_column(self, column):
        if column>= self.w:
            raise(ValueError, "Input is too large.");
        column = [self.g[row][column] for row in range(self.h)];
        return column;
    #############学员自定义函数 end################
        
        
        
    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        # TODO - your code here
        if self.h == 1:
            return self.g[0][0];
        else:
            return self.g[0][0]*self.g[1][1]-self.g[0][1]*self.g[1][0];

    def trace(self):#矩阵轨迹 即对角线和
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        sum = 0;
        for i in range(self.h):
            sum+=self.g[i][i];
        return sum;

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        # TODO - your code here
        if self.h == 1:
            inverse = [[1/self.g[0][0]]];
        else:
            a = self.g[0][0];
            b = self.g[0][1];
            c = self.g[1][0];
            d = self.g[1][1];
            if(a*d==b*c):
                raise ValueError('matrix does not have a inverse!');
            else:
                weigh = 1/(a*d-b*c);
                inverse = [[weigh*d,weigh*-1*b],[weigh*-1*c,weigh*a]];
        return Matrix(inverse);

    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        # TODO - your code here
        matrix_transpose = [];
        
        for j in range(self.w):
            matrix_transpose.append(self.get_column(j));
    
        return Matrix(matrix_transpose);
        
        
    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        #   
        # TODO - your code here
        #
        result = [];
        for i in range(self.h):
            result.append([a+b for a,b in zip(self.g[i],other.g[i])]);
            
        return Matrix(result);
                

        

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #   
        # TODO - your code here
        #
        result = [];
        for row in self.g:
            result.append([-1*n for n in row]);
            
        return Matrix(result);
        
    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #   
        # TODO - your code here
        #
        result = [];
        for i in range(self.h):
            result.append([a-b for a,b in zip(self.g[i],other.g[i])]);
            
        return Matrix(result);
        
    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #   
        # TODO - your code here
        #
        
        result = [];
        row_result = [];
        product = 0;
        
        if(self.w != other.h):
            raise(ValueError, "Matrices can not multiply for their dimesion doesn't match"); 
            
        for row in self.g:
            row_result = [];
            for j in range(other.w):
                product = dot_product(row,other.get_column(j));
                row_result.append(product);
            result.append(row_result);
        
        return Matrix(result);

    def __rmul__(self, other):#标量乘法
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            pass
            #   
            # TODO - your code here
            #
            result = [];
            row_result = [];
            
            for row in self.g:
                row_result = [m*other for m in row];
                result.append(row_result);
            return Matrix(result);
            
            
            
            
            
            