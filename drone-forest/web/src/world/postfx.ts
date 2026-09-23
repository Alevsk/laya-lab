/**
 * Post-processing chain: scene → subtle bloom (sun disc, nav lights, motes) → vignette →
 * OutputPass (ACES tone mapping + sRGB, reading renderer.toneMappingExposure so the world
 * can pulse exposure on a collision).
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { VignetteShader } from 'three/addons/shaders/VignetteShader.js';

export interface PostFx {
  render(): void;
  setSize(width: number, height: number): void;
  dispose(): void;
}

export function createPostFx(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
): PostFx {
  const size = renderer.getSize(new THREE.Vector2());
  const composer = new EffectComposer(renderer);
  const bloom = new UnrealBloomPass(size, 0.32, 0.55, 0.88);
  const vignette = new ShaderPass(VignetteShader);
  vignette.uniforms['offset']!.value = 0.95;
  vignette.uniforms['darkness']!.value = 0.55;
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(bloom);
  composer.addPass(vignette);
  composer.addPass(new OutputPass());

  return {
    render: () => composer.render(),
    setSize(width, height) {
      composer.setSize(width, height);
      bloom.setSize(width, height);
    },
    dispose: () => composer.dispose(),
  };
}
