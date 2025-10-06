import pandas as pd
import numpy as np
from itertools import takewhile


def get_components(df:pd.DataFrame, demand_col:str, output_row:str, 
                sector_col:str, household_col:str, disposable_row:str):
    '''
    Get the components to use in the model

    :param demand_col: Column giving intermediate demand by sector
    :param output_row: Row giving total output by sector
    :param sector_col: Column giving sector labels
    :param household_col: Column giving spend of citizens
    :param disposable_row: Row giving compensation of employees
    :returns: output vector, sector list, disposable income vector, household spend vector, io matrix
    '''
        
    # Columns containing intermediate demand between sectors
    intermediate_cols = [*takewhile(lambda x: x != demand_col, df.columns)]
    intermediate_cols.remove(sector_col)

    # Columns containing intermediate demand and the total output
    output = df.iloc[np.where(df[sector_col] == output_row)][intermediate_cols]

    # disposable income
    if disposable_row is None:
        disp_inc = None
    else:
        disp_inc = df.iloc[np.where(df[sector_col] == disposable_row)][intermediate_cols]
        disp_inc = disp_inc.to_numpy()

    # Household spend
    if household_col is None:
        households = None
    else:
        households = df[[sector_col, household_col]].iloc[:len(intermediate_cols)]
        households = households[household_col].to_numpy()


    # Create the IO Matrix
    io_matrix = df.iloc[:len(intermediate_cols)][intermediate_cols]

    return  output, intermediate_cols, disp_inc, households, io_matrix


def calculate_technical_coefficients(io_matrix: pd.DataFrame, output: pd.Series) -> np.array:
    '''
    Calculate technical coefficient matrix from io matrix and output array

    :param io_matrix: The input output matrix [A, A]
    :param output: Array of total sector output [A,]
    :return: Technical coeff matrix
    '''

    # Calculate the coefficient matrix -- TYPE 1
    io_matrix = io_matrix.to_numpy()
    B = output.to_numpy()
    A = io_matrix / B

    return A


def leontief_inverse(A:np.array)-> np.array:
    '''
    Calculate leontief inverse mults

    :param A: Coefficient matrix
    :returns: array of multipliers by sector, LI matrix
    '''
    # Calculate technology matrices
    I = np.identity(A.shape[0])
    mults_all = np.linalg.inv(I - A) # Leontieff inverse

    return mults_all


def t1_multipliers(io_matrix:np.array, output:np.array) -> tuple[np.array, np.array]:
    '''
    Calculate type 1 Leontief Inverse multipliers

    :param io_matrix: Square IO matrix [A,A]
    :param output: Total output vector [A,]
    :return: Type 1 multipliers, LI matrix
    '''

    # Calc tech coefficients
    A = calculate_technical_coefficients(io_matrix, output)

    # Calculate LI Mults
    mults_all = leontief_inverse(A)

    t1 = np.sum(mults_all, axis=0)

    return t1, mults_all


def t2_multipliers(io_matrix:np.array, output:np.array, households:np.array, disp_inc:np.array, GDHI:float=None) -> tuple[np.array, np.array]:
    '''
    Calculate type 2 Leontief Inverse multipliers

    :param io_matrix: Square IO matrix [A,A]
    :param output: Total output vector [A,]
    :param households: Household spend vector [A,]
    :param disp_inc: Disposable income vector [A,]
    :param GDHI: Total Gross Domestic Household Income constant, default to sum(output)
    :return: Type 2 multipliers, LI Matrix
    '''

    if GDHI is None:
        GDHI = np.sum(output)

    # Calculate the technical coefficients
    A = calculate_technical_coefficients(io_matrix, output)

    # Add row for relative sector income per output
    B = output.to_numpy()
    dinc_per_unit_demand = disp_inc / B
    A_extended = np.vstack([A, dinc_per_unit_demand])

    # Add col for relative sector spend
    households = households / GDHI # scale based on GDHI constant
    spend_column = np.append(households, 0.0).reshape(-1, 1) # household spend on households = 0
    A_type2 = np.hstack([A_extended, spend_column])

    # Calculate multipliers
    mults_all_before_clip = leontief_inverse(A_type2)

    mults_all = mults_all_before_clip[:A.shape[0],:A.shape[0]]
    t2 = np.sum(mults_all, axis=0)

    return t2, mults_all


def model_scenario(mults_all:np.array, demand:pd.Series) -> np.array:
    '''
    Model a scenario based on multipliers and demand

    :param mults: Leontief Multiplier matrix
    :param demand: Demand vector
    :return: Modified demand vector based on the multipliers
    '''
    D = demand.to_numpy().flatten()

    return np.dot(D, mults_all.T)
