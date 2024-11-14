from utils import Calculation
from sympy import symbols
if __name__ == "__main__":
    calculation = Calculation()
    D,R,D0,R0 = symbols("D,R,D0,R0")
    D,R,D0,R0 = calculation.CalcuDR()

    E = float(input("Energy:"))
    U = float(input("Height of the SquareBarrier:"))
    a_value = float(input("Width of this Barrier:"))
    D,R = calculation.CalcuNum(E,U,a_value)