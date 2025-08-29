import React from "react";
import SegmentAndDetectLayout from "./SegmentAndDetectLayout";

const App: React.FC = () => {
  return (
    <div style={{ fontFamily: "Arial, sans-serif", background: "#f5f5f5", minHeight: "100vh" }}>
      
      <main style={{ padding: "2rem" }}>
        <SegmentAndDetectLayout />
      </main>

    </div>
  );
};

export default App;
