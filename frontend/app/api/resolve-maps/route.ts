import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const url = req.nextUrl.searchParams.get("url");
  if (!url) {
    return NextResponse.json({ error: "Missing url query parameter" }, { status: 400 });
  }

  try {
    // Follow HTTP redirects (e.g. https://maps.app.goo.gl/...)
    const resp = await fetch(url, {
      redirect: "follow",
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      },
    });

    const finalUrl = resp.url || url;

    // Pattern extractions for Google Maps destination URLs
    const patterns = [
      /search\/(-?\d+\.\d+),\+?(-?\d+\.\d+)/i,
      /@(-?\d+\.\d+),(-?\d+\.\d+)/i,
      /[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)/i,
      /place\/(-?\d+\.\d+),(-?\d+\.\d+)/i,
      /ll=(-?\d+\.\d+),(-?\d+\.\d+)/i,
    ];

    for (const p of patterns) {
      const match = finalUrl.match(p);
      if (match) {
        const lat = parseFloat(match[1]);
        const lon = parseFloat(match[2]);
        return NextResponse.json({
          status: "ok",
          latitude: lat,
          longitude: lon,
          destination_url: finalUrl,
        });
      }
    }

    return NextResponse.json(
      { error: "Could not find coordinates in resolved Google Maps destination URL", finalUrl },
      { status: 422 }
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: `Failed to follow redirect: ${msg}` }, { status: 500 });
  }
}
