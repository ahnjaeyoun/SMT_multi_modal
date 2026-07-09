import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import RealTime from "./pages/realTime/RealTime";
import DefectManage from "./pages/defectManage/DefectManage";
import ErrorCheck from "./pages/errorCheck/ErrorCheck";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/realtime" replace />} />
          <Route path="/realtime" element={<RealTime />} />
          <Route path="/defect"   element={<DefectManage />} />
          <Route path="/error"    element={<ErrorCheck />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
