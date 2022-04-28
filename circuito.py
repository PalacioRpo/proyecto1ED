"""Problema simple del agente viajero (TSP) en una placa de circuito."""


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import numpy as np
import time

import math
def crear_modelo_datos(inicio):
    """Almacena los datos del problema de circuitos."""
    data = {}
    # ubicacioens de los punto del circuito
    data['lugares'] = [
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 157), (246, 157),
        (236, 169), (228, 169), (228, 161), (220, 169), (212, 169), (204, 169),
        (196, 169), (188, 169), (196, 161), (188, 145), (172, 145), (164, 145),
        (156, 145), (148, 145), (140, 145), (148, 169), (164, 169), (172, 169),
        (156, 169), (140, 169), (132, 169), (124, 169), (116, 161), (104, 153),
        (104, 161), (104, 169), (90, 165), (80, 157), (64, 157), (64, 165),
        (56, 169), (56, 161), (56, 153), (56, 145), (56, 137), (56, 129),
        (56, 121), (40, 121), (40, 129), (40, 137), (40, 145), (40, 153),
        (40, 161), (40, 169), (32, 169), (32, 161), (32, 153), (32, 145),
        (32, 137), (32, 129), (32, 121), (32, 113), (40, 113), (56, 113),
        (56, 105), (48, 99), (40, 99), (32, 97), (32, 89), (24, 89),
        (16, 97), (16, 109), (8, 109), (8, 97), (8, 89), (8, 81),
        (8, 73), (8, 65), (8, 57), (16, 57), (8, 49), (8, 41),
        (24, 45), (32, 41), (32, 49), (32, 57), (32, 65), (32, 73),
        (32, 81), (40, 83), (40, 73), (40, 63), (40, 51), (44, 43),
        (44, 35), (44, 27), (32, 25), (24, 25), (16, 25), (16, 17),
        (24, 17), (32, 17), (44, 11), (56, 9), (56, 17), (56, 25),
        (56, 33), (56, 41), (64, 41), (72, 41), (72, 49), (56, 49),
        (48, 51), (56, 57), (56, 65), (48, 63), (48, 73), (56, 73),
        (56, 81), (48, 83), (56, 89), (56, 97), (104, 97), (104, 105),
        (104, 113), (104, 121), (104, 129), (104, 137), (104, 145), (116, 145),
        (124, 145), (132, 145), (132, 137), (140, 137), (148, 137), (156, 137),
        (164, 137), (172, 125), (172, 117), (172, 109), (172, 101), (172, 93),
        (172, 85), (180, 85), (180, 77), (180, 69), (180, 61), (180, 53),
        (172, 53), (172, 61), (172, 69), (172, 77), (164, 81), (148, 85),
        (124, 85), (124, 93), (124, 109), (124, 125), (124, 117), (124, 101),
        (104, 89), (104, 81), (104, 73), (104, 65), (104, 49), (104, 41),
        (104, 33), (104, 25), (104, 17), (92, 9), (80, 9), (72, 9),
        (64, 21), (72, 25), (80, 25), (80, 25), (80, 41), (88, 49),
        (104, 57), (124, 69), (124, 77), (132, 81), (140, 65), (132, 61),
        (124, 61), (124, 53), (124, 45), (124, 37), (124, 29), (132, 21),
        (124, 21), (120, 9), (128, 9), (136, 9), (148, 9), (162, 9),
        (156, 25), (172, 21), (180, 21), (180, 29), (172, 29), (172, 37),
        (172, 45), (180, 45), (180, 37), (188, 41), (196, 49), (204, 57),
        (212, 65), (220, 73), (228, 69), (228, 77), (236, 77), (236, 69),
        (236, 61), (228, 61), (228, 53), (236, 53), (236, 45), (228, 45),
        (228, 37), (236, 37), (236, 29), (228, 29), (228, 21), (236, 21),
        (252, 21), (260, 29), (260, 37), (260, 45), (260, 53), (260, 61),
        (260, 69), (260, 77), (276, 77), (276, 69), (276, 61), (276, 53),
        (284, 53), (284, 61), (284, 69), (284, 77), (284, 85), (284, 93),
        (284, 101), (288, 109), (280, 109), (276, 101), (276, 93), (276, 85),
        (268, 97), (260, 109), (252, 101), (260, 93), (260, 85), (236, 85),
        (228, 85), (228, 93), (236, 93), (236, 101), (228, 101), (228, 109),
        (228, 117), (228, 125), (220, 125), (212, 117), (204, 109), (196, 101),
        (188, 93), (180, 93), (180, 101), (180, 109), (180, 117), (180, 125),
        (196, 145), (204, 145), (212, 145), (220, 145), (228, 145), (236, 145),
        (246, 141), (252, 125), (260, 129), (280, 133)
    ]  
    data['actuador'] = 1  #punta con soldadura o broca
    data['deposito'] = inicio  #punto según el indice
    return data


def calcular_distancia_euclidiana(lugares):
    """calcular las distancias entre los puntos en un plano usando la distancia euclidiana"""
    distancias = {}
    for desde_lugar, desde_nodo in enumerate(lugares):
        distancias[desde_lugar] = {}
        for para_lugar, para_nodo in enumerate(lugares):
            if desde_lugar == para_lugar:
                distancias[desde_lugar][para_lugar] = 0
            else:
                # distancia Euclidiana
                distancias[desde_lugar][para_lugar] = (int(
                    math.hypot((desde_nodo[0] - para_nodo[0]),
                                (desde_nodo[1] - para_nodo[1]))))
    return distancias


def mostrar_solucion(controlador, enrutamiento, solucion, datos):
    """Muestra la solución por terminal."""
    print('Se logra el objetivo con: {} de costo'.format(solucion.ObjectiveValue()))
    index = enrutamiento.Start(0)
    plan_output = 'Ruta a seguir:\n'
    distancia_enrutamiento = 0
    arr_x = []
    arr_y = []
    while not enrutamiento.IsEnd(index):
        plan_output += ' data[lugar:{}] ->'.format(datos['lugares'][controlador.IndexToNode(index)])
        arr_x.append(datos['lugares'][controlador.IndexToNode(index)][0])
        arr_y.append(datos['lugares'][controlador.IndexToNode(index)][1])
        previous_index = index
        index = solucion.Value(enrutamiento.NextVar(index))
        distancia_enrutamiento += enrutamiento.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(controlador.IndexToNode(index))
    print(plan_output)
    #print("x: ", arr_x, "\n", "y:", arr_y)
    plan_output += 'Objective: {}m\n'.format(distancia_enrutamiento)
    return arr_x, arr_y

'''def mejorSolucion(controlador, enrutamiento, solucion, inicio, datos, mejor, peor):
    if (inicio == len(datos['lugares'])):
        return mejor, peor
    else:
        '''

def trazado(arr_x, arr_y, string):
    fig, ax = plt.subplots()
    if(string == "mejor"):
        ax.set_title("Mejor Solucion")
    else:
        ax.set_title("Peor Solucion")
    ax.plot(arr_x, arr_y, color='green')
    ax.scatter(arr_x, arr_y) 
    plt.show()
    
def mainCompleto():
    mejor = 10000000
    mejor_datos = []
    peor = 0
    peor_datos = []
    memoria_total = 0
    memoria_mejor = 0
    memoria_peor = 0
    tiempoini = time.time()
    i = 0
    while (i < 280):
        '''i = int(input())
        if (i < 0):
            break'''
        '''EIngresar red al programa.'''
        # instanciar el problema.
        data = crear_modelo_datos(i)
        
        # Crear el controlador de índices de enrutamiento
        controlador = pywrapcp.RoutingIndexManager(len(data['lugares']),
                                                    data['actuador'], data['deposito'])
        
        # Crear modelo de enrutamiento.
        enrutamiento = pywrapcp.RoutingModel(controlador)
        
        distance_matrix = calcular_distancia_euclidiana(data['lugares'])
        
        def distance_callback(from_index, to_index):
            '''Calculo de distancia entre dos nodos.'''
            # Convertir de índice de variable de enrutamiento a matriz de distancia NodeIndex.
            desde_nodo = controlador.IndexToNode(from_index)
            para_nodo = controlador.IndexToNode(to_index)
            return distance_matrix[desde_nodo][para_nodo]

        transit_callback_index = enrutamiento.RegisterTransitCallback(distance_callback)
        
        # Definir el costo de cada arista.
        enrutamiento.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Configuración de la heurística de la primera solución.
        parametros_busqueda = pywrapcp.DefaultRoutingSearchParameters()
        parametros_busqueda.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Solución del problema.
        solucion = enrutamiento.SolveWithParameters(parametros_busqueda)
        memoria_total += pywrapcp.Solver_MemoryUsage()
        if (solucion.ObjectiveValue() < mejor):
            mejor = solucion.ObjectiveValue()
            memoria_mejor = pywrapcp.Solver_MemoryUsage()
            mejor_datos = [controlador, enrutamiento, solucion, data]
        if (solucion.ObjectiveValue() > peor):
            peor = solucion.ObjectiveValue()
            memoria_peor = pywrapcp.Solver_MemoryUsage()
            peor_datos = [controlador, enrutamiento, solucion, data]
        i+=1
    
    # Imprimir la solución.
    if solucion:
        print("*"*20, "mejor solucion", "*"*20)
        print("Uso de memoria de la mejor solucion : ", memoria_mejor, "bytes")
        arr_x1, arr_y1 = mostrar_solucion(mejor_datos[0], mejor_datos[1], mejor_datos[2], mejor_datos[3])
        print("*"*20, "peor solucion", "*"*20)
        print("Uso de memoria de la peor solucion : ", memoria_peor, "bytes")
        arr_x2, arr_y2 = mostrar_solucion(peor_datos[0], peor_datos[1], peor_datos[2], peor_datos[3])
        tiempofin = time.time()
        print("*"*20, "Gasto en memoria y tiempo", "*"*20)
        print("gasto en memoria total:", memoria_total, "bytes", '\n', "gasto en tiempo", tiempofin - tiempoini, "s" )
    trazado(arr_x1, arr_y1, "mejor")
    trazado(arr_x2, arr_y2, "peor")
    
def mainDosPuntos():
    mejor = 10000000
    mejor_datos = []
    peor = 0
    peor_datos = []
    memoria_total = 0
    memoria_mejor = 0
    memoria_peor = 0
    tiempoini = time.time()
    i = 0
    while (True):
        print("Ingrese el inicio")
        i = int(input())
        if (i < 0):
            break
        '''EIngresar red al programa.'''
        # instanciar el problema.
        data = crear_modelo_datos(i)
        
        # Crear el controlador de índices de enrutamiento
        controlador = pywrapcp.RoutingIndexManager(len(data['lugares']),
                                                    data['actuador'], data['deposito'])
        
        # Crear modelo de enrutamiento.
        enrutamiento = pywrapcp.RoutingModel(controlador)
        
        distance_matrix = calcular_distancia_euclidiana(data['lugares'])
        
        def distance_callback(from_index, to_index):
            '''Calculo de distancia entre dos nodos.'''
            # Convertir de índice de variable de enrutamiento a matriz de distancia NodeIndex.
            desde_nodo = controlador.IndexToNode(from_index)
            para_nodo = controlador.IndexToNode(to_index)
            return distance_matrix[desde_nodo][para_nodo]

        transit_callback_index = enrutamiento.RegisterTransitCallback(distance_callback)
        
        # Definir el costo de cada arista.
        enrutamiento.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Configuración de la heurística de la primera solución.
        parametros_busqueda = pywrapcp.DefaultRoutingSearchParameters()
        parametros_busqueda.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Solución del problema.
        solucion = enrutamiento.SolveWithParameters(parametros_busqueda)
        memoria_total += pywrapcp.Solver_MemoryUsage()
        if (solucion.ObjectiveValue() < mejor):
            mejor = solucion.ObjectiveValue()
            memoria_mejor = pywrapcp.Solver_MemoryUsage()
            mejor_datos = [controlador, enrutamiento, solucion, data]
        if (solucion.ObjectiveValue() > peor):
            peor = solucion.ObjectiveValue()
            memoria_peor = pywrapcp.Solver_MemoryUsage()
            peor_datos = [controlador, enrutamiento, solucion, data]
    
    # Imprimir la solución.
    if solucion:
        print("*"*20, "mejor solucion", "*"*20)
        print("Uso de memoria de la mejor solucion : ", memoria_mejor, "bytes")
        arr_x1, arr_y1 = mostrar_solucion(mejor_datos[0], mejor_datos[1], mejor_datos[2], mejor_datos[3])
        print("*"*20, "peor solucion", "*"*20)
        print("Uso de memoria de la peor solucion : ", memoria_peor, "bytes")
        arr_x2, arr_y2 = mostrar_solucion(peor_datos[0], peor_datos[1], peor_datos[2], peor_datos[3])
        tiempofin = time.time()
        print("*"*20, "Gasto en memoria y tiempo", "*"*20)
        print("gasto en memoria total:", memoria_total, "bytes", '\n', "gasto en tiempo", tiempofin - tiempoini, "s" )
    trazado(arr_x1, arr_y1, "mejor")
    trazado(arr_x2, arr_y2, "peor")

def main():
    print("Ingrese 1 para proceso completo", '\n', "Ingrese 2 para tomar mejor y peor na mas")
    i = int(input())
    if i == 1:
        mainCompleto()
    if i == 2:
        mainDosPuntos()
    print("Fin Proceso")


if __name__ == '__main__':
    main()