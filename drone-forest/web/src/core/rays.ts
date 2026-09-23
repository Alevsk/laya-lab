/** The fixed ray fan. Must match `server/schemas.py::RAY_SPEC` exactly, in order. */
export const RAY_MAX_RANGE = 60;

export const RAY_SPEC: ReadonlyArray<readonly [name: string, bearingDeg: number, elevationDeg: number]> = [
  ['forward', 0, 0],
  ['left_15', -15, 0],
  ['right_15', 15, 0],
  ['left_30', -30, 0],
  ['right_30', 30, 0],
  ['left_45', -45, 0],
  ['right_45', 45, 0],
  ['left_60', -60, 0],
  ['right_60', 60, 0],
  ['up', 0, 35],
  ['down', 0, -35],
] as const;
