import re

import pandas as pd
import sympy as sp
from tabulate import tabulate


def read_problem(filename: str):
    with open(filename, 'r') as f:
        problem = f.readlines()
        problem = list(map(lambda line: line.strip().replace(" ", ""), problem))
        return problem


def print_problem(z_expr, goal, constraints):
    print(f"Z = {z_expr} -> {goal}")
    for constraint in constraints:
        print(constraint)


def parse_problem(problem_lines: list):
    z_lines = [l for l in problem_lines if '->' in l and re.search(r'\bZ\b', l, re.IGNORECASE)]
    if not z_lines:
        raise ValueError("Не найдена целевая функция Z")
    z_line = z_lines[0]
    problem_lines.remove(z_line)

    z_expr_str, goal = re.match(r'Z\s*=\s*(.+?)\s*->\s*(min|max)', z_line, re.IGNORECASE).groups()
    z_expr = sp.sympify(z_expr_str)

    constraints = []
    for l in problem_lines:
        if '>=' in l:
            lhs, rhs = l.split('>=')
            constraints.append(sp.sympify(lhs.strip()) >= sp.sympify(rhs.strip()))
        elif '<=' in l:
            lhs, rhs = l.split('<=')
            constraints.append(sp.sympify(lhs.strip()) <= sp.sympify(rhs.strip()))
        elif '=' in l:
            lhs, rhs = l.split('=')
            constraints.append(sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip())))
        else:
            raise ValueError(f"Неизвестный формат: {l}")

    variables = sorted(
        z_expr.free_symbols.union(*[
            c.lhs.free_symbols.union(c.rhs.free_symbols) for c in constraints
        ]),
        key=lambda s: s.name
    )
    return z_expr, goal.lower(), constraints, variables


def canonize_problem(z_expr: str, goal: str, constraints: list):
    z_expr = sp.sympify(z_expr)
    if goal not in ('min', 'max'):
        raise ValueError("Цель Z должна быть 'min' или 'max'")

    if goal == 'max':
        z_expr = -z_expr

    canonized_constraints = []

    artificial_var_i = 1
    artificial_variables = []

    for eq in constraints:
        new_var = sp.symbols(f's{artificial_var_i}')

        if isinstance(eq, sp.GreaterThan):
            # x1 + x2 >= a ===> -x1 - x2 + sN = a
            canonized_eq = sp.Eq(-eq.lhs + new_var, -eq.rhs)  # eq.lhs - new_var
            artificial_var_i += 1
            artificial_variables.append(new_var)
        elif isinstance(eq, sp.LessThan):
            # x1 + x2 <= a  ===>  x1 + x2 + sN = a
            canonized_eq = sp.Eq(eq.lhs + new_var, eq.rhs)
            artificial_var_i += 1
            artificial_variables.append(new_var)
        elif isinstance(eq, sp.Equality):
            # x1 + x2 = a
            canonized_eq = eq
        else:
            raise ValueError(f"Неизвестный тип ограничения: {eq}")

        canonized_constraints.append(canonized_eq)

    return z_expr, canonized_constraints, artificial_variables


def build_simplex_table(z_expr, canonized_constraints: list, variables: list, artificial_variables: list):
    all_variables = variables + artificial_variables
    column_headers = [str(v) for v in all_variables] + ['Св. член']

    rows = []
    basis = []
    for eq in canonized_constraints:
        lhs = eq.lhs
        rhs = eq.rhs

        artificial_in_expr = [var for var in artificial_variables if lhs.has(var)]
        if artificial_in_expr:
            basis_variable = artificial_in_expr[0]
        else:
            basis_variable = None

        row = []
        for var in all_variables:
            coeff = lhs.coeff(var)
            row.append(coeff)

        row.append(rhs)
        rows.append(row)
        basis.append(str(basis_variable) if basis_variable is not None else "-")

    z_row = []
    for var in all_variables:
        z_row.append(z_expr.coeff(var))
    z_row.append(0)
    rows.append(z_row)
    basis.append('Z')

    df = pd.DataFrame(rows, columns=column_headers)
    df.insert(0, "Базис", pd.Series(basis))
    return df


def print_simplex_table(df: pd.DataFrame):
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))


def dual_simplex_method(df: pd.DataFrame):
    ...


def main():
    problem = read_problem("in.txt")
    z_expr, goal, constraints, variables = parse_problem(problem)
    print("Исходная проблема:")
    print_problem(z_expr, goal, constraints)
    z_expr, canonized_constraints, artificial_variables = canonize_problem(z_expr, goal, constraints)
    df = build_simplex_table(z_expr, canonized_constraints, variables, artificial_variables)
    print("Cимплекс-таблица:")
    print_simplex_table(df)
    df = dual_simplex_method(df)
    print("Итоговая симплекс-таблица")
    print_simplex_table(df)


if __name__ == "__main__":
    main()
