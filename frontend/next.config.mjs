/** @type {import('next').NextConfig} */
const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://43.204.114.28:8000";

const nextConfig = {
  poweredByHeader: false,
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
