export interface RoomConfig {
  id: string;
  name: string;
  persona: string;
  colors: {
    wall: string;
    floor: string;
    accent: string;
    desk: string;
    chair: string;
  };
  lighting: {
    ambient: number;
    directional: {
      position: [number, number, number];
      intensity: number;
      color: string;
    };
    window: {
      color: string;
      intensity: number;
    };
    secondary?: {
      color: string;
      intensity: number;
      position: [number, number, number];
    };
  };
  agent: {
    outfit: {
      skin: string;
      hair: string;
      hoodie: string;
      hoodieDark: string;
      pants: string;
      shoes: string;
    };
    posture: 'upright' | 'forward' | 'relaxed' | 'still' | 'slouched';
  };
  environment: {
    layout: 'minimal' | 'cyberpunk' | 'creative' | 'analytical' | 'hacker';
    props: string[];
    geometry: {
      floorSize: [number, number];
      wallHeight: number;
      shape: 'rectangle' | 'L' | 'alcove';
    };
    camera: {
      position: [number, number, number];
      target: [number, number, number];
      fov: number;
    };
  };
  behavior: {
    speedMultiplier: number;
    traits: string[];
    linguisticStyle: string;
  };
}

export const ROOMS: RoomConfig[] = [
  {
    id: 'architect',
    name: 'The Architect',
    persona: 'Calm, precise, system-level thinker',
    colors: {
      wall: '#f3f4f6',
      floor: '#d1d5db',
      accent: '#3b82f6',
      desk: '#ffffff',
      chair: '#1f2937',
    },
    lighting: {
      ambient: 0.6,
      directional: {
        position: [5, 10, 5],
        intensity: 1.0,
        color: '#ffffff',
      },
      window: {
        color: '#ffffff',
        intensity: 2,
      },
    },
    agent: {
      outfit: {
        skin: '#f3ccb0',
        hair: '#4b5563',
        hoodie: '#f9fafb',
        hoodieDark: '#f3f4f6',
        pants: '#4b5563',
        shoes: '#1f2937',
      },
      posture: 'upright',
    },
      environment: {
        layout: 'minimal',
        props: ['ultrawide_monitor', 'schematic_panels'],
        geometry: {
          floorSize: [12, 10],
          wallHeight: 4,
          shape: 'rectangle',
        },
        camera: {
          position: [8, 5, 8],
          target: [0, 0.8, 0],
          fov: 45,
        },
      },
    behavior: {
      speedMultiplier: 1.0,
      traits: ['logical', 'composed', 'methodical'],
      linguisticStyle: 'Precise, formal, and structured. Focuses on systems and architecture.',
    },
  },
  {
    id: 'operator',
    name: 'The Operator',
    persona: 'Fast, aggressive, always-on',
    colors: {
      wall: '#1e293b',
      floor: '#0f172a',
      accent: '#a855f7',
      desk: '#334155',
      chair: '#1e293b',
    },
    lighting: {
      ambient: 0.5,
      directional: {
        position: [2, 5, 2],
        intensity: 1.2,
        color: '#e9d5ff',
      },
      window: {
        color: '#a855f7',
        intensity: 5,
      },
      secondary: {
        color: '#22d3ee',
        intensity: 4,
        position: [-2, 2, -2],
      },
    },
    agent: {
      outfit: {
        skin: '#f3ccb0',
        hair: '#111827',
        hoodie: '#1e293b',
        hoodieDark: '#0f172a',
        pants: '#1e293b',
        shoes: '#7c3aed',
      },
      posture: 'forward',
    },
      environment: {
        layout: 'cyberpunk',
        props: ['holo_displays', 'server_racks'],
        geometry: {
          floorSize: [10, 8],
          wallHeight: 3.5,
          shape: 'rectangle',
        },
        camera: {
          position: [6, 3, 6],
          target: [0, 0.5, 0],
          fov: 60,
        },
      },
    behavior: {
      speedMultiplier: 1.5,
      traits: ['proactive', 'alert', 'decisive'],
      linguisticStyle: 'Concise, action-oriented, and urgent. Uses tactical terminology.',
    },
  },
  {
    id: 'creator',
    name: 'The Creator',
    persona: 'Curious, playful, imaginative',
    colors: {
      wall: '#fdf6e3',
      floor: '#eee8d5',
      accent: '#cb4b16',
      desk: '#d4a373',
      chair: '#586e75',
    },
    lighting: {
      ambient: 0.7,
      directional: {
        position: [3, 6, 3],
        intensity: 1.2,
        color: '#ffedd5',
      },
      window: {
        color: '#fef3c7',
        intensity: 3,
      },
    },
    agent: {
      outfit: {
        skin: '#f3ccb0',
        hair: '#b45309',
        hoodie: '#cb4b16',
        hoodieDark: '#9a3412',
        pants: '#586e75',
        shoes: '#268bd2',
      },
      posture: 'relaxed',
    },
      environment: {
        layout: 'creative',
        props: ['plants', 'sketches', 'tablet'],
        geometry: {
          floorSize: [11, 10],
          wallHeight: 3.5,
          shape: 'L',
        },
        camera: {
          position: [7, 4.5, 7],
          target: [0, 0.8, 0],
          fov: 50,
        },
      },
    behavior: {
      speedMultiplier: 0.8,
      traits: ['imaginative', 'flexible', 'exploratory'],
      linguisticStyle: 'Creative, metaphorical, and slightly informal. Encourages experimentation.',
    },
  },
  {
    id: 'analyst',
    name: 'The Analyst',
    persona: 'Logical, observant, data-driven',
    colors: {
      wall: '#f8fafc',
      floor: '#e2e8f0',
      accent: '#0891b2',
      desk: '#f1f5f9',
      chair: '#334155',
    },
    lighting: {
      ambient: 0.5,
      directional: {
        position: [0, 10, 0],
        intensity: 1.5,
        color: '#f0f9ff',
      },
      window: {
        color: '#e0f2fe',
        intensity: 2,
      },
    },
    agent: {
      outfit: {
        skin: '#f3ccb0',
        hair: '#1e293b',
        hoodie: '#0891b2',
        hoodieDark: '#0e7490',
        pants: '#334155',
        shoes: '#020617',
      },
      posture: 'still',
    },
      environment: {
        layout: 'analytical',
        props: ['dashboards', 'graphs', 'grid'],
        geometry: {
          floorSize: [14, 8],
          wallHeight: 3.5,
          shape: 'rectangle',
        },
        camera: {
          position: [0, 8, 10],
          target: [0, 0, 0],
          fov: 40,
        },
      },
    behavior: {
      speedMultiplier: 0.9,
      traits: ['observant', 'analytical', 'objective'],
      linguisticStyle: 'Fact-based, detailed, and analytical. Uses data points and logic.',
    },
  },
  {
    id: 'hacker',
    name: 'The Hacker',
    persona: 'Obsessive, nocturnal, experimental',
    colors: {
      wall: '#0f172a',
      floor: '#020617',
      accent: '#22c55e',
      desk: '#1e293b',
      chair: '#0f172a',
    },
    lighting: {
      ambient: 0.35,
      directional: {
        position: [0, 4, 2],
        intensity: 0.8,
        color: '#86efac',
      },
      window: {
        color: '#22c55e',
        intensity: 3,
      },
      secondary: {
        color: '#22c55e',
        intensity: 5,
        position: [0, 1.5, -1.5], // Screen glow
      },
    },
    agent: {
      outfit: {
        skin: '#f3ccb0',
        hair: '#020617',
        hoodie: '#020617',
        hoodieDark: '#000000',
        pants: '#020617',
        shoes: '#1e293b',
      },
      posture: 'slouched',
    },
      environment: {
        layout: 'hacker',
        props: ['cables', 'terminal_monitors', 'city_lights'],
        geometry: {
          floorSize: [12, 12],
          wallHeight: 4,
          shape: 'alcove',
        },
        camera: {
          position: [6, 4, 6],
          target: [0, 0.5, -0.5],
          fov: 70,
        },
      },
    behavior: {
      speedMultiplier: 1.2,
      traits: ['obsessive', 'unconventional', 'tenacious'],
      linguisticStyle: 'Terse, technical, and often uses slang or terminal metaphors.',
    },
  },
];
