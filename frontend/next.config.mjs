/** @type {import('next').NextConfig} */
const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "https://deep-learning-based-super-resolution-mapping-srm-production.up.railway.app";

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
