/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/analyze',
        destination: 'http://127.0.0.1:8000/analyze',
      },
      {
        source: '/api/weather',
        destination: 'http://127.0.0.1:8000/weather',
      },
      {
        source: '/api/health',
        destination: 'http://127.0.0.1:8000/health',
      },
      {
        source: '/api/export-health-report-pdf',
        destination: 'http://127.0.0.1:8000/api/export-health-report-pdf',
      },
    ];
  },
};

export default nextConfig;
