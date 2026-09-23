/**
 * The quadcopter mesh, built from primitives so the game ships with no external assets.
 * `root` carries the position; `body` is a child that takes the visual bank/pitch so the
 * kinematics never have to un-rotate anything. Rotors and nav-light materials are exposed
 * for animation. All meshes sit on SCENERY_LAYER so the drone never occludes its own sensors.
 */
import * as THREE from 'three';
import { markScenery } from '../world/layers';

export interface DroneModel {
  readonly root: THREE.Group;
  readonly body: THREE.Group;
  readonly rotors: THREE.Object3D[];
  readonly navLeft: THREE.MeshStandardMaterial;
  readonly navRight: THREE.MeshStandardMaterial;
  readonly strobe: THREE.MeshStandardMaterial;
  dispose(): void;
}

const ARM_LENGTH = 0.62;
const ARM_ANGLES = [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4];

function navLight(color: number): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: 0x111111,
    emissive: color,
    emissiveIntensity: 0,
    roughness: 0.3,
  });
}

export function buildDroneModel(): DroneModel {
  const root = new THREE.Group();
  root.name = 'drone';
  const body = new THREE.Group();
  root.add(body);

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];
  const track = <G extends THREE.BufferGeometry>(g: G): G => {
    geometries.push(g);
    return g;
  };
  const mat = <M extends THREE.Material>(m: M): M => {
    materials.push(m);
    return m;
  };

  const shell = mat(new THREE.MeshStandardMaterial({ color: 0x2b2f36, metalness: 0.55, roughness: 0.42 }));
  const accent = mat(new THREE.MeshStandardMaterial({ color: 0xd9822b, metalness: 0.3, roughness: 0.5 }));
  const carbon = mat(new THREE.MeshStandardMaterial({ color: 0x15171b, metalness: 0.7, roughness: 0.35 }));
  const bladeMat = mat(
    new THREE.MeshStandardMaterial({ color: 0x0c0d10, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.9 }),
  );
  const discMat = mat(
    new THREE.MeshBasicMaterial({ color: 0x9aa3ad, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }),
  );
  const lensMat = mat(new THREE.MeshStandardMaterial({ color: 0x0a1a2a, metalness: 0.9, roughness: 0.1 }));

  const hull = new THREE.Mesh(track(new THREE.BoxGeometry(0.46, 0.14, 0.62, 1, 1, 1)), shell);
  body.add(hull);
  const canopy = new THREE.Mesh(track(new THREE.CapsuleGeometry(0.13, 0.32, 4, 10)), accent);
  canopy.rotation.x = Math.PI / 2;
  canopy.position.y = 0.1;
  body.add(canopy);
  const battery = new THREE.Mesh(track(new THREE.BoxGeometry(0.28, 0.1, 0.34)), carbon);
  battery.position.y = -0.11;
  body.add(battery);

  const pod = new THREE.Mesh(track(new THREE.CylinderGeometry(0.06, 0.06, 0.12, 12)), carbon);
  pod.rotation.x = Math.PI / 2;
  pod.position.set(0, -0.04, -0.36);
  body.add(pod);
  const lens = new THREE.Mesh(track(new THREE.SphereGeometry(0.045, 12, 8)), lensMat);
  lens.position.set(0, -0.04, -0.43);
  body.add(lens);

  const armGeo = track(new THREE.BoxGeometry(ARM_LENGTH, 0.04, 0.06));
  const motorGeo = track(new THREE.CylinderGeometry(0.06, 0.07, 0.09, 12));
  const bladeGeo = track(new THREE.BoxGeometry(0.5, 0.008, 0.045));
  const discGeo = track(new THREE.CircleGeometry(0.26, 24));
  const legGeo = track(new THREE.CylinderGeometry(0.012, 0.012, 0.16, 6));
  const lightGeo = track(new THREE.SphereGeometry(0.03, 10, 8));

  const navLeft = mat(navLight(0xff2a1a));
  const navRight = mat(navLight(0x2dff5a));
  const strobe = mat(navLight(0xffffff));

  const rotors: THREE.Object3D[] = [];
  for (const a of ARM_ANGLES) {
    const arm = new THREE.Mesh(armGeo, carbon);
    arm.position.set((Math.cos(a) * ARM_LENGTH) / 2, 0, (Math.sin(a) * ARM_LENGTH) / 2);
    arm.rotation.y = -a;
    body.add(arm);

    const tip = new THREE.Vector3(Math.cos(a) * ARM_LENGTH, 0, Math.sin(a) * ARM_LENGTH);
    const motor = new THREE.Mesh(motorGeo, shell);
    motor.position.copy(tip).setY(0.05);
    body.add(motor);

    const rotor = new THREE.Group();
    rotor.position.copy(tip).setY(0.11);
    const bladeA = new THREE.Mesh(bladeGeo, bladeMat);
    const bladeB = new THREE.Mesh(bladeGeo, bladeMat);
    bladeB.rotation.y = Math.PI / 2;
    rotor.add(bladeA, bladeB);
    const disc = new THREE.Mesh(discGeo, discMat);
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = -0.005;
    rotor.add(disc);
    body.add(rotor);
    rotors.push(rotor);

    const leg = new THREE.Mesh(legGeo, carbon);
    leg.position.copy(tip).multiplyScalar(0.55).setY(-0.15);
    body.add(leg);

    // front arms (a in the -Z half) carry the red/green wingtip lights, rear arms the strobe
    const front = Math.sin(a) < 0;
    const light = new THREE.Mesh(lightGeo, front ? (Math.cos(a) < 0 ? navLeft : navRight) : strobe);
    light.position.copy(tip).setY(-0.01);
    body.add(light);
  }

  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      o.castShadow = true;
      o.receiveShadow = false;
    }
  });
  markScenery(root);

  return {
    root,
    body,
    rotors,
    navLeft,
    navRight,
    strobe,
    dispose() {
      for (const g of geometries) g.dispose();
      for (const m of materials) m.dispose();
    },
  };
}
