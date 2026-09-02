"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

type SpaceSceneProps = {
  compact?: boolean;
  className?: string;
};

type SatelliteRuntime = {
  root: THREE.Group;
  craft: THREE.Group;
  beam?: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  radius: number;
  speed: number;
  phase: number;
};

function createEarthTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1536;
  canvas.height = 768;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return new THREE.CanvasTexture(canvas);
  }

  const ocean = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  ocean.addColorStop(0, "#062a56");
  ocean.addColorStop(0.45, "#0b4971");
  ocean.addColorStop(1, "#031224");
  ctx.fillStyle = ocean;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.globalAlpha = 0.82;
  const continents = [
    { x: 230, y: 245, sx: 150, sy: 190, r: -0.24 },
    { x: 330, y: 470, sx: 88, sy: 190, r: 0.08 },
    { x: 650, y: 250, sx: 135, sy: 145, r: 0.08 },
    { x: 820, y: 255, sx: 260, sy: 170, r: -0.05 },
    { x: 1010, y: 430, sx: 96, sy: 140, r: 0.4 },
    { x: 1190, y: 375, sx: 115, sy: 92, r: -0.12 },
    { x: 1390, y: 260, sx: 130, sy: 170, r: -0.22 },
  ];

  for (const land of continents) {
    ctx.save();
    ctx.translate(land.x, land.y);
    ctx.rotate(land.r);
    const landGradient = ctx.createRadialGradient(0, 0, 10, 0, 0, Math.max(land.sx, land.sy));
    landGradient.addColorStop(0, "#4e7c55");
    landGradient.addColorStop(0.62, "#315c44");
    landGradient.addColorStop(1, "#1b332b");
    ctx.fillStyle = landGradient;
    ctx.beginPath();
    for (let i = 0; i < 28; i += 1) {
      const a = (Math.PI * 2 * i) / 28;
      const wobble = 0.82 + Math.sin(i * 2.13) * 0.1 + Math.cos(i * 1.31) * 0.08;
      const px = Math.cos(a) * land.sx * wobble;
      const py = Math.sin(a) * land.sy * (0.8 + Math.cos(i * 1.7) * 0.08);
      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  ctx.globalAlpha = 0.32;
  ctx.strokeStyle = "#a9d8aa";
  ctx.lineWidth = 1.2;
  for (let i = 0; i < 72; i += 1) {
    const x = (i * 211) % canvas.width;
    const y = 82 + ((i * 97) % 610);
    ctx.beginPath();
    ctx.ellipse(x, y, 22 + (i % 5) * 8, 6 + (i % 4) * 3, i * 0.19, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.globalAlpha = 0.74;
  ctx.fillStyle = "rgba(255, 214, 130, 0.9)";
  for (let i = 0; i < 520; i += 1) {
    const cluster = continents[i % continents.length];
    const x = cluster.x + (Math.sin(i * 12.989) * 0.5 + 0.5 - 0.5) * cluster.sx * 1.45;
    const y = cluster.y + (Math.cos(i * 78.233) * 0.5 + 0.5 - 0.5) * cluster.sy * 1.25;
    const size = i % 7 === 0 ? 1.5 : 0.8;
    ctx.fillRect(x, y, size, size);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function createCloudTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return new THREE.CanvasTexture(canvas);
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < 86; i += 1) {
    const x = (i * 137) % canvas.width;
    const y = 40 + ((i * 71) % 430);
    const gradient = ctx.createRadialGradient(x, y, 4, x, y, 42 + (i % 7) * 10);
    gradient.addColorStop(0, "rgba(255,255,255,0.42)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.ellipse(x, y, 90 + (i % 5) * 24, 18 + (i % 6) * 5, i * 0.21, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createSatelliteModel(kind: number): THREE.Group {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: kind === 1 ? "#d7e7ef" : "#a8b5c7",
    metalness: 0.72,
    roughness: 0.28,
    emissive: "#061525",
  });
  const panelMaterial = new THREE.MeshStandardMaterial({
    color: "#193d76",
    emissive: "#006b9a",
    emissiveIntensity: 0.42,
    metalness: 0.38,
    roughness: 0.3,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: "#111827",
    metalness: 0.65,
    roughness: 0.34,
  });

  const body = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.22, 0.24), bodyMaterial);
  group.add(body);

  const sensor = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, 0.14, 18), darkMaterial);
  sensor.rotation.x = Math.PI / 2;
  sensor.position.z = 0.2;
  group.add(sensor);

  const panelWidth = kind === 2 ? 0.42 : 0.72;
  const panelLeft = new THREE.Mesh(new THREE.BoxGeometry(panelWidth, 0.03, 0.18), panelMaterial);
  panelLeft.position.x = -0.55;
  const panelRight = panelLeft.clone();
  panelRight.position.x = 0.55;
  group.add(panelLeft, panelRight);

  if (kind !== 2) {
    const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.32, 8), darkMaterial);
    antenna.rotation.z = Math.PI / 2.7;
    antenna.position.y = 0.18;
    antenna.position.x = 0.08;
    group.add(antenna);
  }

  group.scale.setScalar(kind === 2 ? 0.68 : 0.88);
  return group;
}

function createOrbit(radius: number, color: string): THREE.LineLoop {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < 192; i += 1) {
    const a = (i / 192) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.33,
    depthWrite: false,
  });
  return new THREE.LineLoop(geometry, material);
}

export default function SpaceScene({ compact = false, className = "" }: SpaceSceneProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050814, 0.022);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(compact ? 0 : -0.55, compact ? 2.1 : 1.6, compact ? 6.4 : 7.8);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6));
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0x6f85b8, 0.62);
    const sun = new THREE.DirectionalLight(0xffffff, 2.25);
    sun.position.set(4.8, 2.4, 5.4);
    const rim = new THREE.PointLight(0x00d9ff, 1.3, 12);
    rim.position.set(-3.6, 1.2, 2.2);
    scene.add(ambient, sun, rim);

    const earthGroup = new THREE.Group();
    earthGroup.rotation.z = -0.22;
    scene.add(earthGroup);

    const earthTexture = createEarthTexture();
    const cloudTexture = createCloudTexture();
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(compact ? 1.35 : 1.82, 96, 96),
      new THREE.MeshStandardMaterial({
        map: earthTexture,
        roughness: 0.9,
        metalness: 0.02,
        emissive: new THREE.Color("#06182f"),
        emissiveIntensity: 0.08,
      }),
    );
    earthGroup.add(earth);

    const clouds = new THREE.Mesh(
      new THREE.SphereGeometry((compact ? 1.35 : 1.82) * 1.014, 96, 96),
      new THREE.MeshStandardMaterial({
        map: cloudTexture,
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
      }),
    );
    earthGroup.add(clouds);

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry((compact ? 1.35 : 1.82) * 1.045, 96, 96),
      new THREE.MeshBasicMaterial({
        color: "#00d9ff",
        transparent: true,
        opacity: 0.11,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      }),
    );
    earthGroup.add(atmosphere);

    const starsGeometry = new THREE.BufferGeometry();
    const starCount = compact ? 900 : 1700;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i += 1) {
      const radius = 18 + Math.random() * 46;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = radius * Math.cos(phi);
    }
    starsGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(
      starsGeometry,
      new THREE.PointsMaterial({
        color: "#d8f7ff",
        size: compact ? 0.018 : 0.022,
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
      }),
    );
    scene.add(stars);

    const satellites: SatelliteRuntime[] = [];
    const orbitConfigs = compact
      ? [
          { radius: 2.15, speed: 0.34, phase: 0.2, rot: [0.42, 0.14, -0.18], color: "#00d9ff", kind: 0, beam: true },
          { radius: 2.58, speed: -0.24, phase: 2.5, rot: [-0.18, 0.24, 0.38], color: "#6c63ff", kind: 2 },
        ]
      : [
          { radius: 2.58, speed: 0.32, phase: 0.25, rot: [0.5, 0.18, -0.16], color: "#00d9ff", kind: 0, beam: true },
          { radius: 3.04, speed: -0.22, phase: 1.9, rot: [-0.26, 0.38, 0.32], color: "#6c63ff", kind: 1 },
          { radius: 3.48, speed: 0.16, phase: 3.3, rot: [0.12, -0.44, 0.52], color: "#32e875", kind: 2 },
          { radius: 4.02, speed: -0.11, phase: 5.1, rot: [-0.42, -0.12, -0.36], color: "#00d9ff", kind: 0 },
        ];

    for (const config of orbitConfigs) {
      const root = new THREE.Group();
      root.rotation.set(config.rot[0], config.rot[1], config.rot[2]);
      const orbit = createOrbit(config.radius, config.color);
      root.add(orbit);

      const craft = createSatelliteModel(config.kind);
      root.add(craft);

      let beam: SatelliteRuntime["beam"];
      if (config.beam) {
        beam = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
          new THREE.LineBasicMaterial({
            color: "#00d9ff",
            transparent: true,
            opacity: 0.46,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
          }),
        );
        root.add(beam);
      }

      scene.add(root);
      satellites.push({
        root,
        craft,
        beam,
        radius: config.radius,
        speed: config.speed,
        phase: config.phase,
      });
    }

    const pointer = new THREE.Vector2(0, 0);
    const onPointerMove = (event: PointerEvent) => {
      const bounds = mount.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
      pointer.y = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
    };
    mount.addEventListener("pointermove", onPointerMove);

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);

    let animationFrame = 0;
    const clock = new THREE.Clock();
    const animate = () => {
      const elapsed = clock.getElapsedTime();
      earth.rotation.y = elapsed * 0.055;
      clouds.rotation.y = elapsed * 0.073;
      stars.rotation.y = elapsed * 0.004;

      for (const satellite of satellites) {
        const angle = satellite.phase + elapsed * satellite.speed;
        satellite.craft.position.set(Math.cos(angle) * satellite.radius, Math.sin(angle * 1.35) * 0.12, Math.sin(angle) * satellite.radius);
        satellite.craft.lookAt(0, 0, 0);
        satellite.craft.rotateY(Math.PI);
        if (satellite.beam) {
          satellite.beam.geometry.setFromPoints([satellite.craft.position.clone(), new THREE.Vector3(0.28, 0.18, 0.4)]);
          satellite.beam.material.opacity = 0.24 + Math.sin(elapsed * 2.2) * 0.12;
        }
      }

      camera.position.x += ((compact ? 0 : -0.55) + pointer.x * 0.18 - camera.position.x) * 0.035;
      camera.position.y += ((compact ? 2.1 : 1.6) - pointer.y * 0.1 - camera.position.y) * 0.035;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      mount.removeEventListener("pointermove", onPointerMove);
      mount.removeChild(renderer.domElement);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Points) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose());
          } else {
            material.dispose();
          }
        }
      });
      earthTexture.dispose();
      cloudTexture.dispose();
      renderer.dispose();
    };
  }, [compact]);

  return <div ref={mountRef} className={`absolute inset-0 ${className}`} aria-hidden="true" data-testid="space-scene" />;
}
