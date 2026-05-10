import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Totti's WINNER Lab",
  description: "Jリーグ スコア予想アプリ",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
