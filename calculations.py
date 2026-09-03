import argparse, os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

#para el manejo de incertidumbres
from uncertainties import ufloat
from uncertainties.umath import *



"""
variables de entrada del programa, para cada gota, en el archivo:

volt: voltaje, en volts.
ionization_level: id asignado a cada gota, para calcular el promedio de la carga y guardarlo en un array.
t_fall: tiempo de caida libre, en segundos.
t_field: tiempo de subida o bajada con campo, en segundos.
field_direction: dirección del campo: por defecto es 1 que indica que la gota sube con el campo, si es -1 baja, e indica su signo a la hora de calcular el campo

formato del archivo:
volt,ionization_level,t_fall,t_field, field_direction
"""

#variable debug
debug_mode = False
debug_lite = False

#CONSTANTES

#Se usa una incertidumbre de 0.05 mm para las distancias, tomando en cuenta que la grid tiene una resolución de 0.1 mm
TRAVEL_DISTANCE = ufloat(5e-4,5e-5) #en m, medio milímetro se recorre
PLATE_DISTANCE = ufloat(76e-4,5e-5) #en m, el manual describe el espaciador de plástico aproximadamente con 7.6 mm de espesor

OIL_DENSITY = 886 #en kg/m^3
GRAVITY = 9.81 #en m/s^2
B = 82e-4 #en Pa*m, constante de Cunningham


#VARIABLES EXPERIMENTALES QUE SE MANTUVIERON CONSTANTES
RESISTANCE = ufloat(2e6, 2000)  # en ohmios
BAROMETIC_PRESSURE = ufloat(768e2, 2e3)  # Pascales

pi = np.pi #pi

# Umbral para MCD
MCD_TOLERANCE = 0.15

# FPS del video
FPS = 30

#funciones

def DropTemperature(Resistance):
    """
    Hace una interpolación (cúbica) de la tabla del apéndice B para obtener la temperatura en base a la resistencia.
    """
    #datos de la tabla
    table_celsius = list(range(10,40)) #°C
    table_ohms = [
    3.239e6, 3.118e6, 3.004e6, 2.897e6, 2.795e6, 2.700e6, 2.610e6, 2.526e6, 2.446e6, 2.371e6,
    2.300e6, 2.233e6, 2.169e6, 2.110e6, 2.053e6, 2.000e6, 1.950e6, 1.902e6, 1.857e6, 1.815e6,
    1.774e6, 1.736e6, 1.700e6, 1.666e6, 1.634e6, 1.603e6, 1.574e6, 1.547e6, 1.521e6, 1.496e6
    ] #ohm
    
    #interpolar temperatura con la resistencia usando la tabla
    interpolation = interp1d(table_ohms,table_celsius, kind='cubic', bounds_error=False, fill_value=(10.0, 39.0))
    
    #si Resistance es un ufloat, se extrae su valor nominal para interpolar
    if hasattr(Resistance, 'nominal_value'):
        temp_nominal = float(interpolation(Resistance.nominal_value))
        #para propagar la incertidumbre, se evalua la interpolación con Resistance ± su incertidumbre
        delta = Resistance.std_dev
        if delta > 0:
            temp_plus = float(interpolation(Resistance.nominal_value + delta))
            temp_minus = float(interpolation(Resistance.nominal_value - delta))
            #se toma la mayor desviación como incertidumbre
            temp_std = max(abs(temp_plus - temp_nominal), abs(temp_minus - temp_nominal))
            return ufloat(temp_nominal, temp_std)
        return ufloat(temp_nominal, 0)
    else:
        #si es un float normal, se le asigna una incertidumbre pequeña por la interpolación
        return ufloat(float(interpolation(Resistance)), 0.1)

def AirViscosity(Temp):
    """
    Hace una interpolación cubica de la gráfica del apéndice A para obtener la viscosidad del aire en base a la temperatura.
    """
    
    #datos de la gráfica
    graph_celsius = [
        15.0000000000, 15.9947807360, 16.9912771900, 17.9894893630, 18.9894172530,
        19.9893451430, 20.9858415970, 21.9840537690, 22.9805502230, 23.9502292600,
        24.9518728680, 25.9518007580, 26.9448657760, 27.9430779480, 28.9127569850,
        29.9982842820, 30.9982121720, 32.0015714980
    ]  # °C

    graph_viscosities = [
        1.8000753856e-5, 1.8048119219e-5, 1.8095234091e-5, 1.8143215933e-5, 1.8190473300e-5,
        1.8238597637e-5, 1.8285712508e-5, 1.8333836845e-5, 1.8380951717e-5, 1.8426850131e-5,
        1.8474974468e-5, 1.8523098805e-5, 1.8570282674e-5, 1.8617190553e-5, 1.8664305425e-5,
        1.8715483656e-5, 1.8762987014e-5, 1.8808801431e-5
    ]  # N*s/m^2
    
    #interpolar viscosidad
    interpolation = interp1d(graph_celsius, graph_viscosities, kind='cubic', bounds_error=False, fill_value=(15.0, 33.0))
    
    #si Temp es un ufloat, se extrae su valor nominal para interpolar
    if hasattr(Temp, 'nominal_value'):
        visc_nominal = float(interpolation(Temp.nominal_value))
        #para propagar la incertidumbre
        delta = Temp.std_dev
        if delta > 0:
            visc_plus = float(interpolation(Temp.nominal_value + delta))
            visc_minus = float(interpolation(Temp.nominal_value - delta))
            visc_std = max(abs(visc_plus - visc_nominal), abs(visc_minus - visc_nominal))
            return ufloat(visc_nominal, visc_std)
        return ufloat(visc_nominal, 0)
    else:
        return ufloat(float(interpolation(Temp)), 1e-8)
    
def FallVelocity(FallTime):
    """
    Calcula la velocidad de caida libre, tomando en cuenta que se registra el tiempo que le toma en recorrer 0.5 mm.
    """
    return TRAVEL_DISTANCE/FallTime #m/s

def FieldVelocity(FieldTime):
    """
    Calcula la velocidad de subida o bajada con campo presente, tomando en cuenta que se registra el tiempo que le toma en recorrer 0.5 mm.
    """
    return TRAVEL_DISTANCE/FieldTime #m/s
    
def DropRadius(AirViscosity,Vfall,BarometricPressure):
    """
    Usando la ecuación (12) del manual, se calcula el radio de la gota.
    """
    radius = sqrt(pow(B/(2*BarometricPressure),2)+((9*AirViscosity*Vfall)/(2*OIL_DENSITY*GRAVITY)))-(B/(2*BarometricPressure)) #m
    return radius #en el manual su simbolo es a.
    
def DropMass(Radius):
    """
    Usando la ecuación (4) del manual, se calcula la masa de la gota.
    """
    mass = (pi*4/3)*pow(Radius,3)*OIL_DENSITY #kg
    return mass

def DropCharge(Volt, Mass, Vfall, Vfield, FieldDirection=1):
    """
    Usando la ecuación (3a) del manual, se calcula la carga de la gota, cambiando el signo de la velocidad de caida libre dependiendo si la gota sube o baja.
    """
    charge = (Mass*GRAVITY*PLATE_DISTANCE*(Vfall+FieldDirection*Vfield))/(Volt*Vfall) #C
    return charge

def ReadTimeWithUncertainty(time_value):
    """
    Convierte un tiempo medido a ufloat con incertidumbre.
    Para video a FPS fijos, la incertidumbre típica es 1/(2*FPS)
    """
    uncertainty = 1.0 / (2.0 * FPS)
    return ufloat(time_value, uncertainty)

# -----------------------------------------------------------------------------
# MCD APROXIMADO CON DIFERENCIAS
# -----------------------------------------------------------------------------

def gcd_approx_differences(scaled_charges, tolerance=MCD_TOLERANCE):
    """
    Calcula el MCD aproximado usando el método de diferencias.
    """
    # Extraer valores nominales
    nominal_values = [abs(c.nominal_value) if hasattr(c, 'nominal_value') else abs(c) for c in scaled_charges]
    
    # Filtrar valores cero o muy pequeños
    nominal_values = [v for v in nominal_values if v > 1e-10]
    
    if len(nominal_values) < 2:
        return None
    
    nominal_values.sort()
    
    # Función para verificar si a es aproximadamente múltiplo de b
    def is_multiple(a, b, tol):
        if b == 0:
            return False
        ratio = a / b
        nearest = round(ratio)
        if nearest == 0:
            return False
        return abs(ratio - nearest) / nearest < tol
    
    # Algoritmo de Euclides aproximado
    def gcd_approx(a, b, tol):
        if b == 0:
            return a
        
        if a < b:
            a, b = b, a
        
        if is_multiple(a, b, tol):
            return b
        
        if b / a < tol:
            return b
        
        q = round(a / b)
        r = a - q * b
        
        if r < tol * b:
            return b
        
        return gcd_approx(b, r, tol)
    
    # Calcular diferencias entre todas las cargas
    charge_diffs = []
    for i in range(len(nominal_values)):
        for j in range(i+1, len(nominal_values)):
            diff = abs(nominal_values[j] - nominal_values[i])
            if diff > 1e-10:
                charge_diffs.append(diff)
    
    if not charge_diffs:
        return None
    
    # Calcular MCD de todas las diferencias
    result = charge_diffs[0]
    for i in range(1, len(charge_diffs)):
        result = gcd_approx(result, charge_diffs[i], tolerance)
        if result <= 1:
            break
    
    # Verificar si el MCD es 1 (problema)
    if result <= 1.5 and (debug_lite or debug_mode):
        print(f"  ⚠️ ADVERTENCIA: MCD = {result:.2f} (prácticamente 1). Las cargas no son múltiplos consistentes.")
        print(f"  Cargas escaladas: {nominal_values}")
    
    return result

def CalculateE(Charges, ScaleFactor=1e22):
    """
    Usando las cargas de todas las gotas del mismo nivel de ionización y voltaje,
    calcula las diferencias de las cargas, obtiene el MCD aproximado,
    y con ello la carga del electrón.
    """
    # Escalar las cargas
    scaled_charges = [c * ScaleFactor for c in Charges]
    
    if debug_lite or debug_mode:
        if all(hasattr(c, 'nominal_value') for c in scaled_charges):
            print("Cargas escaladas (nominales): " + str([round(c.nominal_value) for c in scaled_charges]) + ".")
            print("Incertidumbres: " + str([c.std_dev for c in scaled_charges]) + ".")
        else:
            print("Cargas escaladas: " + str([round(c) for c in scaled_charges]) + ".")
    
    # Calcular MCD usando diferencias
    e_scaled = gcd_approx_differences(scaled_charges, MCD_TOLERANCE)
    
    if e_scaled is None or e_scaled <= 0:
        if debug_lite or debug_mode:
            print("No se pudo encontrar un MCD válido.")
        return ufloat(float('nan'), float('nan'))
    
    # Calcular el error de e a partir de las incertidumbres de las cargas
    if all(hasattr(c, 'std_dev') for c in scaled_charges):
        std_devs = [c.std_dev for c in scaled_charges]
        if len(std_devs) > 0 and sum(std_devs) > 0:
            e_std = np.sqrt(sum(s**2 for s in std_devs)) / len(scaled_charges) / ScaleFactor
        else:
            e_std = e_scaled / ScaleFactor * 0.05
    else:
        e_std = e_scaled / ScaleFactor * 0.05
    
    e_nom_val = e_scaled / ScaleFactor
    if e_std < e_nom_val * 0.01:
        e_std = e_nom_val * 0.01
    elif e_std > e_nom_val * 0.5:
        e_std = e_nom_val * 0.5
    
    e = ufloat(e_nom_val, e_std)
    
    if debug_lite or debug_mode:
        print(f"Carga del electrón calculada: {e} C")
    
    return e

def DropChargeSameIonizationLevel(DropMeditions):
    """
    Calcula el promedio de todas las cargas de la gota en la misma run (voltaje y nivel de ionización)
    """
    #cargas que suben
    up_charges = []
    
    #cargas que bajan 
    down_charges = []
    
    #para el debug_mode
    volt = 0.0
    ion_lvl = 0
    
    #variable para medir cual medición de las 10 hay
    med_count = 1.0
    
    #itera en el DataFrame
    for _, medition in DropMeditions.iterrows(): 
    
        ion_lvl = int(medition['ionization_level'])
    
        #variables obtenidas de la medición
        volt = ufloat(medition['volt'], 0.1)  #incertidumbre pequeña en el voltaje
        t_fall = ReadTimeWithUncertainty(medition['t_fall'])
        t_field = ReadTimeWithUncertainty(medition['t_field'])
        field_direction = medition['field_direction']
        
        if field_direction == 1:
            direc = "subiendo"
        else:
            direc = "bajando"
        
        #variables constantes en todas las mediciones
        resistance = RESISTANCE
        barometric_pressure = BAROMETIC_PRESSURE
        
        #variables calculadas con las funciones
        temperature = DropTemperature(resistance)
        
        if debug_mode:
            print("---------------------------\nTemperatura de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(temperature)+"°C.")
        
        viscosity = AirViscosity(temperature)
        
        if debug_mode:
            print("Viscosidad del aire de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(viscosity)+" N*s/m^2.")
        
        v_fall = FallVelocity(t_fall)
        
        if debug_mode:
            print("Velocidad de caída de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(v_fall)+" m/s.")
        
        v_field = FieldVelocity(t_field)
        
        if debug_mode:
            if field_direction == 1:
                print("Velocidad de ascenso de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(v_field)+" m/s, en presencia del campo.")
            else:
                print("Velocidad de descenso de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(v_field)+" m/s, en presencia del campo.")
            
        
        radius = DropRadius(viscosity,v_fall,barometric_pressure)
        
        if debug_mode:
            print("Radio de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(radius)+" m.")
        
        mass = DropMass(radius)
        
        if debug_mode:
            print("Masa de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(mass)+" kg.")
        
        #se calcula la carga y añade al arreglo de mediciones para la gota
        q = DropCharge(volt,mass,v_fall,v_field,field_direction)
        
        if debug_mode:
            print("Carga de la gota "+str(int(med_count))+" ionizada "+str(ion_lvl) +" veces a "+str(volt)+" V, mientras está "+direc+": "+str(q)+" C.\n")
        
        if field_direction == 1:
            up_charges.append(q)
        else:
            down_charges.append(q)
        
        med_count+=1.0
    
    # Calcular promedios
    if up_charges:
        up_mean = np.mean(up_charges)
        if hasattr(up_mean, 'nominal_value'):
            up_charge_avrg = ufloat(abs(up_mean.nominal_value), up_mean.std_dev)
        else:
            up_charge_avrg = abs(up_mean)
    else:
        up_charge_avrg = ufloat(0, 0)
    
    if down_charges:
        down_mean = np.mean(down_charges)
        if hasattr(down_mean, 'nominal_value'):
            down_charge_avrg = ufloat(abs(down_mean.nominal_value), down_mean.std_dev)
        else:
            down_charge_avrg = abs(down_mean)
    else:
        down_charge_avrg = ufloat(0, 0)
    
    # Verificar si las cargas positivas y negativas son muy diferentes
    up_val = up_charge_avrg.nominal_value if hasattr(up_charge_avrg, 'nominal_value') else up_charge_avrg
    down_val = down_charge_avrg.nominal_value if hasattr(down_charge_avrg, 'nominal_value') else down_charge_avrg
    
    if up_val > 0 and down_val > 0:
        ratio = max(up_val, down_val) / min(up_val, down_val)
        if ratio > 1.5 and (debug_lite or debug_mode):
            print(f"  Discrepancia entre cargas promedio:")
            print(f"    Subiendo: {up_charge_avrg} C")
            print(f"    Bajando: {down_charge_avrg} C")
    
    if debug_lite:
        print("Carga promedio de la gotas subiendo, ionizadas "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(up_charge_avrg)+" C, en presencia del campo.\n")
        print("Carga promedio de la gotas bajando, ionizadas "+str(ion_lvl) +" veces a "+str(volt)+" V: "+str(down_charge_avrg)+" C, en presencia del campo.\n")
    
    charge = (up_charge_avrg + down_charge_avrg) / 2
        
    return charge

def ProcessCharges(volt_data):
    """
    Procesa cada medición para calcular las cargas en cada nivel de ionización
    """
    
    #crea un arreglo vacío donde van a ir las cargas de cada gota
    charges=[]
    
    #separa las mediciones en cada nivel de ionización (A = 1, B = 2, C = 3)
    volt_groups = volt_data.groupby('ionization_level')
    for ion,group in volt_groups:
        run_charge = DropChargeSameIonizationLevel(group)
        charges.append(run_charge)
    
    return charges

def ProcessE(all_data):
    """
    Procesa cada medición para calcular la carga del electrón en cada potencial,
    """
    
    #crea un arreglo vacío donde van a ir las cargas del electron para cada voltaje
    es=[]
    
    #separa las mediciones en cada gota, primero por voltaje
    all_groups = all_data.groupby('volt')
    for volt,group in all_groups:
        
        if debug_lite or debug_mode:
            print("------------------------------------------\nVoltaje: "+str(volt)+" V:")
        
        #y luego por las veces que se ha ionizado (A = 1, B = 2, C = 3)
        same_ionization_charges = ProcessCharges(group)
        
        if debug_lite or debug_mode:
            print("Cargas promedio con voltaje V = "+str(volt)+" (A, B y C respectivamente): "+str(same_ionization_charges)+".\n")
        
        e = CalculateE(same_ionization_charges)
        
        if debug_lite or debug_mode:
            print("Carga del electrón para "+str(volt)+" V: "+str(e)+".\n\n\n")
            
        es.append(e)
    
    # Promedio simple de todas las cargas del electrón
    # Filtrar valores NaN
    valid_e = [e for e in es if hasattr(e, 'nominal_value') and not np.isnan(e.nominal_value)]
    
    if not valid_e:
        return ufloat(float('nan'), float('nan'))
    
    # Promedio
    if all(hasattr(e, 'nominal_value') for e in valid_e):
        mean_value = np.mean(valid_e)
        nominal_vals = [e.nominal_value for e in valid_e]
        std_of_mean = np.std(nominal_vals) / np.sqrt(len(nominal_vals))
        mean_std = np.mean([e.std_dev for e in valid_e])
        combined_std = np.sqrt(std_of_mean**2 + mean_std**2)
        
        return ufloat(mean_value.nominal_value, combined_std)
    else:
        return np.mean(valid_e)

#Programa para ejecutar
def CalcEFromFile(file, debug="None"):

    #modo debug
    global debug_mode
    global debug_lite
    match debug.upper():
        case "Debug":
            debug_mode = True
        case "Lite":
            debug_lite = True
            
    #lee los datos
    try:
        all_data = pd.read_csv(file, header=None, names=['volt','ionization_level','t_fall','t_field','field_direction'])
    except pd.errors.EmptyDataError:
        print(f"El archivo {file} está vacío. Por favor agrega datos.")
        return ufloat(0, 0)
    
    #procesa todos los datos para obtener las cargas, con ellas calcula la carga del electrón.
    e = ProcessE(all_data)
    if debug_mode or debug_lite:
        print("La carga del electrón calculada es:\n"+str(e)+" C.")
    return e
    

#programa al ejecutar
def main():
    """
    Ejecuta todo el programa para obtener e, leyendo el archivo de entrada para pasarlo al procesador de cargas
    """

    #establece el modo debug
    global debug_mode
    global debug_lite

    #parsea el archivo de datos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-lite', action='store_true')
    parser.add_argument('datos', nargs='?', default='data.csv')
    args = parser.parse_args()
    
    debug_mode = args.debug
    debug_lite = args.debug_lite
    
    if debug_mode:
        print("Modo debug activado")
    elif debug_lite:
        print("Modo debug lite activado")
    
    if not os.path.exists(args.datos):
        with open(args.datos, 'w') as f:
            pass

    #lee los datos
    try:
        all_data = pd.read_csv(args.datos, header=None, names=['volt','ionization_level','t_fall','t_field','field_direction'])
    except pd.errors.EmptyDataError:
        print(f"El archivo {args.datos} está vacío. Por favor agrega datos.")
        return
   
    #procesa todos los datos para obtener las cargas, con ellas calcula la carga del electrón.
    e = ProcessE(all_data)
    nom = e.nominal_value
    std = e.std_dev

    print(f"La carga del electrón calculada es:\n {nom:.2e} ± {std:.2e} C.")

if __name__ == "__main__":
    main()
