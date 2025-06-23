from controller import Supervisor

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# === Vehículo AD0 ===
veh0 = supervisor.getFromDef("SUMO_VEHICLE_AD0")
veh0_posicion = veh0.getField("translation")
vel0_kmph = 12
vel0_mps = vel0_kmph / 3.6
tiempo0_inicio = supervisor.getTime()

# === Vehículo AD1 ===
veh1 = supervisor.getFromDef("SUMO_VEHICLE_AD1")
veh1_posicion = veh1.getField("translation")
vel1_kmph = 10
vel1_mps = vel1_kmph / 3.6
tiempo1_inicio = supervisor.getTime()

# === Peatón estático ===
peaton_estatico = supervisor.getFromDef("PEATON_AD0")
peaton_posicion = peaton_estatico.getField("translation")
tiempo_peaton_inicio = supervisor.getTime()
duracion_peaton_visible = 20.0  # segundos


# Parámetros comunes
direccion = [0, 1, 0]
duracion = 15.0  # segundos

while supervisor.step(timestep) != -1:
    t = supervisor.getTime()
    dt = timestep / 1000.0

    # --- Movimiento vehículo AD0 ---
    if t - tiempo0_inicio < duracion:
        pos0 = veh0_posicion.getSFVec3f()
        nueva_pos0 = [
            pos0[0] + direccion[0] * vel0_mps * dt,
            pos0[1] + direccion[1] * vel0_mps * dt,
            pos0[2] + direccion[2] * vel0_mps * dt
        ]
        veh0_posicion.setSFVec3f(nueva_pos0)
    else:
        veh0_posicion.setSFVec3f([0, 0, -100])

    # --- Movimiento vehículo AD1 ---
    if t - tiempo1_inicio < duracion:
        pos1 = veh1_posicion.getSFVec3f()
        nueva_pos1 = [
            pos1[0] + direccion[0] * vel1_mps * dt,
            pos1[1] + direccion[1] * vel1_mps * dt,
            pos1[2] + direccion[2] * vel1_mps * dt
        ]
        veh1_posicion.setSFVec3f(nueva_pos1)
    else:
        veh1_posicion.setSFVec3f([0, 0, -100])

    # --- Lógica para ocultar peatón estático ---
    if t - tiempo_peaton_inicio >= duracion_peaton_visible:
        peaton_posicion.setSFVec3f([0, -100, 0])