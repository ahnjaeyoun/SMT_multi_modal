export const mockSummary    = { total: 127, pr: 74, sd: 53, defect_rate: 6.9 };

export const mockByType     = {
  items: [
    { type: "미납",       count: 28 },
    { type: "납부족",     count: 21 },
    { type: "납쇼트",     count: 17 },
    { type: "냉납",       count: 14 },
    { type: "밀림",       count: 11 },
    { type: "납볼",       count:  9 },
    { type: "쇼트",       count:  8 },
    { type: "역삽",       count:  7 },
    { type: "브릿지",     count:  5 },
    { type: "납좌표밀림", count:  4 },
    { type: "뒤집힘",     count:  3 },
  ],
};

export const mockCause      = {
  items: [
    { name: "온도",   value: 34.2 },
    { name: "습도",   value: 27.1 },
    { name: "진동",   value: 18.5 },
    { name: "가속도", value: 12.4 },
    { name: "소음",   value:  7.8 },
  ],
};

const DATES_DAILY = ["09-01","09-02","09-03","09-04","09-05","09-06","09-07","09-08","09-09","09-10"];
const DATES_MONTHLY = ["2025-06","2025-07","2025-08","2025-09"];
const DATES_QUARTERLY = ["2025 Q1","2025 Q2","2025 Q3"];
const DATES_YEARLY = ["2023","2024","2025"];

export const mockTrend = {
  daily:     DATES_DAILY.map(d => ({ date: d, PR: Math.floor(Math.random()*10+3), SD: Math.floor(Math.random()*8+2) })),
  monthly:   DATES_MONTHLY.map(d => ({ date: d, PR: Math.floor(Math.random()*80+40), SD: Math.floor(Math.random()*60+30) })),
  quarterly: DATES_QUARTERLY.map(d => ({ date: d, PR: Math.floor(Math.random()*250+120), SD: Math.floor(Math.random()*180+80) })),
  yearly:    DATES_YEARLY.map(d => ({ date: d, PR: Math.floor(Math.random()*900+400), SD: Math.floor(Math.random()*700+300) })),
};

export const mockRecent = {
  items: [
    { id: "1", time: "18:38:38", process: "PR", type: "미납",   confidence: 91.2 },
    { id: "2", time: "18:39:42", process: "SD", type: "냉납",   confidence: 87.4 },
    { id: "3", time: "18:40:05", process: "PR", type: "밀림",   confidence: 76.1 },
    { id: "4", time: "18:41:14", process: "SD", type: "납볼",   confidence: 82.3 },
    { id: "5", time: "18:42:10", process: "PR", type: "납부족", confidence: 94.5 },
    { id: "6", time: "18:43:30", process: "SD", type: "쇼트",   confidence: 88.0 },
    { id: "7", time: "18:44:50", process: "PR", type: "역삽",   confidence: 79.3 },
  ],
};
