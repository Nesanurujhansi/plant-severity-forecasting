import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const payload = await req.json();

    const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000";
    const res = await fetch(`${backendUrl}/api/export-health-report-pdf`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errDetail = await res.text().catch(() => res.statusText);
      return NextResponse.json(
        { error: `Backend PDF generation failed (${res.status}): ${errDetail}` },
        { status: res.status }
      );
    }

    const pdfBuffer = await res.arrayBuffer();
    const disposition = res.headers.get("content-disposition") || 'attachment; filename="Crop_Health_Report.pdf"';

    return new NextResponse(pdfBuffer, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": disposition,
        "Cache-Control": "no-cache, no-store, must-revalidate",
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || "Internal server error generating PDF report." },
      { status: 500 }
    );
  }
}
