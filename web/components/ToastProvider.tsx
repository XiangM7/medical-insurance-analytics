"use client";

import { Toaster } from "react-hot-toast";

export default function ToastProvider() {
  return (
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: "#1a1a1a",
          color: "#e5e5e5",
          border: "1px solid #262626",
          fontSize: "13px",
        },
      }}
    />
  );
}
