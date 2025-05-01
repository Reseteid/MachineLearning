import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    """
    Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы. 
    Для X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]) ответ 3.

    Args:
        matrix (np.array): Прямоугольная матрица
    
    Rerurns:
        float: Произведение ненулевых элементов диагонали матрицы
    """
    temp = matrix.diagonal()
    return temp[temp != 0].prod()


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    """
    Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество. 
    Для x = np.array([1, 2, 2, 4]), y = np.array([4, 2, 1, 2]) ответ True.

    Args:
        x (np.array): Первый вектор
        y (np.array): Второй вектор
    
    Rerurns:
        bool: Задают ли векторы одно и то же мультимножество
    """
    return (np.sort(x) == np.sort(y)).all()


def max_before_zero_vectorized(x: np.array):
    """
    Найти максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой. 
    Для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответ 5.

    Args:
        x (np.array): Вектор элементов
    
    Rerurns:
        float: Максимальный элемент в векторе, среди тех, перед которыми стоит нулевой
    """
    return x[np.where(x[:-1] == 0)[0] + 1].max()


def add_weighted_channels_vectorized(image: np.array):
    """
    Операции с изображением.
    Складывает каналы изображения с указанными весами, и возвращает результат в виде матрицы размера (height, width). 
    Преобразует цветное изображение в оттенки серого, используя коэффициенты np.array([0.299, 0.587, 0.114]).

    Args:
        image (np.array): трёхмерный массив, содержащий изображение, размера (height, width, numChannels), а также вектор длины numChannels.
    
    Rerurns:
        np.array: преобразованное изображение в оттенках серого в виде матрицы размера (height, width).
    """
    gray = np.array([0.299, 0.587, 0.114])
    return np.dot(image.reshape(image.shape[0]**2, 3), gray).reshape(image.shape[0], image.shape[0])


def run_length_encoding_vectorized(x: np.array):
    """
    Кодирование длин серий (Run-length encoding).
    Возвращает кортеж из двух векторов одинаковой длины. Первый содержит числа, а второй - сколько раз их нужно повторить. Пример: x = np.array([2, 2, 2, 3, 3, 3, 5, 2, 2]). Ответ: (np.array([2, 3, 5, 2]), np.array([3, 3, 1, 2])).

    Args:
        x (np.array): вектор np.array
    
    Rerurns:
        (np.array, np.array): кортеж из двух векторов одинаковой длины. Первый содержит числа, а второй - сколько раз их нужно повторить.
    """
    y = x.copy()
    temp = np.where(np.concatenate((np.array([y[0]]), y), axis=0)[:-1] != x)[0]
    one = np.concatenate((np.array([x[0]]), x[temp]), axis=0)
    two = np.diff(np.concatenate((np.array([0]), temp, np.array([x.size])), axis=0))
    return (one, two)
