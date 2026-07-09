import Navbar from "./Navbar";
import Sidebar from "./Sidebar";

export default function Layout({ children }) {
  return (
    <div style={s.root}>
      <Navbar />
      <div style={s.body}>
        <Sidebar />
        <div style={s.content}>{children}</div>
      </div>
    </div>
  );
}

const s = {
  root:    { display: "flex", flexDirection: "column", height: "100vh", background: "#0f172a", color: "#fff", fontFamily: "sans-serif" },
  body:    { display: "flex", flex: 1, overflow: "hidden" },
  content: { flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" },
};
