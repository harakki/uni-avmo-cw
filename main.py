import re
from fractions import Fraction

import pandas as pd
import sympy as sp
from rich import box
from rich.console import Console
from rich.table import Table


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


def print_simplex_table(df: pd.DataFrame, pivot_row=None, pivot_col=None, ratios=None):
    console = Console(force_terminal=True)
    table = Table(show_header=True, box=box.MARKDOWN)

    headers = list(df.columns)
    for col in headers:
        table.add_column(str(col), justify="right")

    for i, row in df.iterrows():
        row_cells = []
        for j, col in enumerate(headers):
            val = row[col]
            cell_text = str(val)
            if pivot_row is not None and pivot_col is not None:
                if i == pivot_row and not j == pivot_col + 1:
                    cell_text = f"[cyan]{cell_text}[/cyan]"
                if i == pivot_row and j == pivot_col + 1:
                    cell_text = f"[bold bright_cyan]{cell_text}[/bold bright_cyan]"
            row_cells.append(cell_text)
        table.add_row(*row_cells)

    console.print(table)

    if ratios is not None:
        # abs(Z-строка / разрешающая_строка)
        ratios_text = ", ".join(f"{headers[j + 1]} = {str(ratios[j])}" for j in sorted(ratios.keys()))
        console.print(f"Отношения (для выбора разрешающего столбца): {ratios_text}")


def dual_simplex_method(df: pd.DataFrame):
    numeric_cols = df.columns.drop('Базис')
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: Fraction(str(x)))

    m = len(df) - 1  # Количество ограничений
    free_col = df.columns[-1]  # Колонка "Свободный член"

    step = 0
    while True:
        # Маска отрицательности свободных членов в ограничениях (N, [True|False])
        neg_free = df.loc[df.index[:m], free_col] < 0
        if not neg_free.any():
            break
        step += 1

        # Индекс наименьшего свободного члена
        pivot_row = int(df.loc[df.index[:m], free_col].idxmin())

        pivot_row_coeffs = df.iloc[pivot_row, 1:-1]
        # Индексы отрицательных коэффициентов в разрешающей строке
        negative_pivot_row_coeffs = [i for i, val in enumerate(pivot_row_coeffs) if val < 0]
        if not negative_pivot_row_coeffs:
            print("Задача не имеет решения (нет отрицательных коэффициентов в разрешающей строке).")
            return df

        # Отношения |Z-строка_{j}| / |Разрешающая_строка_{ij}|
        z_row_coeffs = df.iloc[m, 1:-1]
        ratios = {i: abs(z_row_coeffs.iat[i]) / abs(pivot_row_coeffs.iat[i]) for i in negative_pivot_row_coeffs}
        pivot_col = min(ratios, key=ratios.get)  # Перебор словаря по минимальному значению (ratios.get(key))

        print_simplex_table(df, pivot_row=pivot_row, pivot_col=pivot_col, ratios=ratios)

        # Деление разрешающей строки на опорный элемент
        pivot_val = df.iat[pivot_row, pivot_col + 1]
        df.iloc[pivot_row, 1:] = df.iloc[pivot_row, 1:] / pivot_val

        # Обнуление всех коэффициентов в разрешающем столбце исключая разрешающую строку
        for i in df.index:
            if i == pivot_row:
                continue
            factor = df.iat[i, pivot_col + 1]
            df.iloc[i, 1:] = df.iloc[i, 1:] - factor * df.iloc[pivot_row, 1:]

        # Обновление базиса
        new_var = df.columns[pivot_col + 1]
        df.at[pivot_row, 'Базис'] = new_var

    return df


def get_optimal_solution(df: pd.DataFrame):
    headers = list(df.columns)
    all_variables = headers[1:-1]
    m = len(df) - 1  # Количество ограничений

    basis_map = {}
    solution = {var: Fraction(0) for var in all_variables}

    for eq in range(m):
        b = df.iloc[eq]["Базис"]
        if b in all_variables:
            basis_map[eq] = b
            solution[b] = Fraction(str(df.iloc[eq]["Св. член"]))

    optimal_z = Fraction(str(df.iloc[m]["Св. член"]))
    basis = list(basis_map.values())
    non_basis = sorted([v for v in all_variables if v not in basis])

    return optimal_z, solution, all_variables, basis, non_basis


def print_optimal_solution(optimal_z: Fraction, solution: dict):
    console = Console(force_terminal=True)
    vars_str = ", ".join(solution.keys())
    values_str = ", ".join(str(v) for v in solution.values())
    console.print(f"Z = ({vars_str}) = ({values_str}) = {optimal_z}")


def get_general_form_solution(df: pd.DataFrame, basis: list, non_basis: list):
    m = len(df) - 1  # Количество ограничений
    general_solution = {}

    # Выражения для базисных переменных
    basis_row_index = {df.iloc[i]["Базис"]: i for i in range(m) if df.iloc[i]["Базис"] in basis}
    for b, row_index in basis_row_index.items():
        general_solution[b] = build_general_form_solution_rhs_expr(df, row_index, non_basis)

    # Выражения для небазисных переменных
    for v in non_basis:
        general_solution[v] = f"{v}"

    # 3. Выражение для Z
    z_expr = build_general_form_solution_rhs_expr(df, m, non_basis)

    return general_solution, z_expr


def build_general_form_solution_rhs_expr(df: pd.DataFrame, row_index: int, non_basis: list):
    expr_parts = [str(df.iloc[row_index]["Св. член"])]
    for v in non_basis:
        coeff = df.iloc[row_index][v]
        term = format_term(-coeff, v)
        if term:
            expr_parts.append(term)

    return sp.sympify("".join(expr_parts))


def format_term(coeff: Fraction, var: str):
    if coeff == 0:
        return ""

    sign = "+" if coeff > 0 else "-"
    abs_coeff = abs(coeff)

    if abs_coeff == 1:
        term = f"{sign} {var}"
    else:
        term = f"{sign} {abs_coeff} * {var}"
    return term


def print_general_form_solution(solution_expr_dict: dict, z_expr: str, non_basis: list):
    console = Console(force_terminal=True)
    print("\nОбщий вид решения:")
    for var in solution_expr_dict.keys():
        console.print(f"{var} = {solution_expr_dict[var]}")

    console.print(f"Z = {z_expr}")

    if non_basis:
        console.print(", ".join([f"{v} >= 0" for v in non_basis]))


def main():
    problem = read_problem("in.txt")
    z_expr, goal, constraints, variables = parse_problem(problem)
    print("Исходная проблема:")
    print_problem(z_expr, goal, constraints)
    z_expr, canonized_constraints, artificial_variables = canonize_problem(z_expr, goal, constraints)

    df = build_simplex_table(z_expr, canonized_constraints, variables, artificial_variables)
    df = dual_simplex_method(df)
    print_simplex_table(df)

    optimal_z, solution, all_variables, basis, non_basis = get_optimal_solution(df)
    print_optimal_solution(optimal_z, solution)

    general_solution, z_expression = get_general_form_solution(df, basis, non_basis)
    print_general_form_solution(general_solution, z_expression, non_basis)


if __name__ == "__main__":
    main()
