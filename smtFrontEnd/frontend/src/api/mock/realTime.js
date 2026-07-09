const PRODUCTS = [
  { file_name: "PR_DEF_MF_A_20250902-183838_01658", category: "pr", status: "DEF", cause: 17, confidence: 0.91, img_path: null, sensor_data: [] },
  { file_name: "SD_NOR_MF_A_20250902-183901_01659", category: "sd", status: "NOR", cause: null, confidence: 0.98, img_path: null, sensor_data: [] },
  { file_name: "PR_NOR_MF_A_20250902-183920_01660", category: "pr", status: "NOR", cause: null, confidence: 0.95, img_path: null, sensor_data: [] },
  { file_name: "SD_DEF_MF_A_20250902-183942_01661", category: "sd", status: "DEF", cause: 19, confidence: 0.87, img_path: null, sensor_data: [] },
  { file_name: "PR_DEF_MF_A_20250902-184005_01662", category: "pr", status: "DEF", cause: 24, confidence: 0.76, img_path: null, sensor_data: [] },
  { file_name: "SD_NOR_MF_A_20250902-184028_01663", category: "sd", status: "NOR", cause: null, confidence: 0.99, img_path: null, sensor_data: [] },
  { file_name: "PR_NOR_MF_A_20250902-184051_01664", category: "pr", status: "NOR", cause: null, confidence: 0.93, img_path: null, sensor_data: [] },
  { file_name: "SD_DEF_MF_A_20250902-184114_01665", category: "sd", status: "DEF", cause: 20, confidence: 0.82, img_path: null, sensor_data: [] },
];

const SENSOR_ROWS = Array.from({ length: 20 }, (_, i) => ({
  time: `18:3${String(i % 10).padStart(2, "0")}:00`,
  온도: +(22 + Math.sin(i * 0.5) * 2).toFixed(2),
  습도: +(55 + Math.cos(i * 0.4) * 5).toFixed(2),
  진동: +(0.3 + Math.sin(i * 0.8) * 0.1).toFixed(3),
  가속도: +(1.0 + Math.cos(i * 0.6) * 0.2).toFixed(3),
  소음: +(42 + Math.sin(i * 0.3) * 3).toFixed(2),
}));

export async function fetchDashboardStats() {
  return { total: 1842, defect: 127, normal: 1715, avg_accuracy: 93.4, defect_rate: 6.9, per_minute: 12 };
}

export async function fetchProducts() {
  return { items: PRODUCTS };
}

export async function fetchThresholds() {
  return {
    pr: { 온도: { mean: 23.5 }, 습도: { mean: 57.0 }, 진동: { mean: 0.35 }, 가속도: { mean: 1.1 }, 소음: { mean: 44.0 } },
    sd: { 온도: { mean: 24.0 }, 습도: { mean: 58.0 }, 진동: { mean: 0.38 }, 가속도: { mean: 1.2 }, 소음: { mean: 45.0 } },
  };
}

export async function fetchInferenceResult() {
  return { defect_types: ["납부족", "냉납"], accuracy: 0.91 };
}

export async function fetchSensorImportance() {
  return { sensor_importance: { temperature: 0.29, noise: 0.27, humidity: 0.21, vibration: 0.13, acceleration: 0.10 } };
}

export function subscribeRealtimeStream(onMessage) {
  let i = 0;
  const id = setInterval(() => {
    const base = PRODUCTS[i % PRODUCTS.length];
    onMessage({
      file_name: base.file_name,
      category: base.category,
      cause: base.cause,
      confidence: base.confidence,
      sensor_data: SENSOR_ROWS.slice(i % 10, (i % 10) + 5),
    });
    i++;
  }, 2000);
  return { close: () => clearInterval(id) };
}
