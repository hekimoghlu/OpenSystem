/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "DisplayListRecorder.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

namespace DisplayList {

class RecorderImpl : public Recorder {
    WTF_MAKE_TZONE_ALLOCATED(RecorderImpl);
    WTF_MAKE_NONCOPYABLE(RecorderImpl);
public:
    WEBCORE_EXPORT RecorderImpl(DisplayList&, const GraphicsContextState&, const FloatRect& initialClip, const AffineTransform&, const DestinationColorSpace& = DestinationColorSpace::SRGB(), DrawGlyphsMode = DrawGlyphsMode::Normal);
    WEBCORE_EXPORT virtual ~RecorderImpl();

    bool isEmpty() const { return m_displayList.isEmpty(); }

    void save(GraphicsContextState::Purpose) final;
    void restore(GraphicsContextState::Purpose) final;
    void translate(float x, float y) final;
    void rotate(float angle) final;
    void scale(const FloatSize&) final;
    void setCTM(const AffineTransform&) final;
    void concatCTM(const AffineTransform&) final;
    void setLineCap(LineCap) final;
    void setLineDash(const DashArray&, float dashOffset) final;
    void setLineJoin(LineJoin) final;
    void setMiterLimit(float) final;
    void resetClip() final;
    void clip(const FloatRect&) final;
    void clipRoundedRect(const FloatRoundedRect&) final;
    void clipOut(const FloatRect&) final;
    void clipOut(const Path&) final;
    void clipOutRoundedRect(const FloatRoundedRect&) final;
    void clipPath(const Path&, WindRule) final;
    void beginTransparencyLayer(float) final;
    void beginTransparencyLayer(CompositeOperator, BlendMode) final;
    void endTransparencyLayer() final;
    void drawRect(const FloatRect&, float) final;
    void drawLine(const FloatPoint& point1, const FloatPoint& point2) final;
    void drawLinesForText(const FloatPoint&, float thickness, const DashArray& widths, bool printing, bool doubleLines, StrokeStyle) final;
    void drawDotsForDocumentMarker(const FloatRect&, DocumentMarkerLineStyle) final;
    void drawEllipse(const FloatRect&) final;
    void drawPath(const Path&) final;
    void drawFocusRing(const Path&, float outlineWidth, const Color&) final;
    void drawFocusRing(const Vector<FloatRect>&, float outlineOffset, float outlineWidth, const Color&) final;
    void fillEllipse(const FloatRect&) final;
    void fillRect(const FloatRect&, RequiresClipToRect) final;
    void fillRect(const FloatRect&, const Color&) final;
    void fillRect(const FloatRect&, Gradient&) final;
    void fillRect(const FloatRect&, Gradient&, const AffineTransform&, RequiresClipToRect) final;
    void fillRect(const FloatRect&, const Color&, CompositeOperator, BlendMode) final;
    void fillRoundedRect(const FloatRoundedRect&, const Color&, BlendMode) final;
    void fillRectWithRoundedHole(const FloatRect&, const FloatRoundedRect&, const Color&) final;
#if ENABLE(VIDEO)
    void drawVideoFrame(VideoFrame&, const FloatRect& destination, ImageOrientation, bool shouldDiscardAlpha) final;
#endif
    void strokeRect(const FloatRect&, float) final;
    void strokeEllipse(const FloatRect&) final;
    void clearRect(const FloatRect&) final;
    void drawControlPart(ControlPart&, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle&) final;
#if USE(CG)
    void applyStrokePattern() final;
    void applyFillPattern() final;
#endif
    void applyDeviceScaleFactor(float) final;

    void beginPage(const IntSize&) final;
    void endPage() final;

    void setURLForRect(const URL&, const FloatRect&) final;

private:
    void recordSetInlineFillColor(PackedColor::RGBA) final;
    void recordSetInlineStroke(SetInlineStroke&&) final;
    void recordSetState(const GraphicsContextState&) final;
    void recordClearDropShadow() final;
    void recordClipToImageBuffer(ImageBuffer&, const FloatRect& destinationRect) final;
    void recordDrawFilteredImageBuffer(ImageBuffer*, const FloatRect& sourceImageRect, Filter&) final;
    void recordDrawGlyphs(const Font&, std::span<const GlyphBufferGlyph>, std::span<const GlyphBufferAdvance>, const FloatPoint& localAnchor, FontSmoothingMode) final;
    void recordDrawDecomposedGlyphs(const Font&, const DecomposedGlyphs&) final;
    void recordDrawImageBuffer(ImageBuffer&, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions) final;
    void recordDrawNativeImage(RenderingResourceIdentifier imageIdentifier, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions) final;
    void recordDrawSystemImage(SystemImage&, const FloatRect&) final;
    void recordDrawPattern(RenderingResourceIdentifier, const FloatRect& destRect, const FloatRect& tileRect, const AffineTransform&, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;
#if ENABLE(INLINE_PATH_DATA)
    void recordFillLine(const PathDataLine&) final;
    void recordFillArc(const PathArc&) final;
    void recordFillClosedArc(const PathClosedArc&) final;
    void recordFillQuadCurve(const PathDataQuadCurve&) final;
    void recordFillBezierCurve(const PathDataBezierCurve&) final;
#endif
    void recordFillPathSegment(const PathSegment&) final;
    void recordFillPath(const Path&) final;
#if ENABLE(INLINE_PATH_DATA)
    void recordStrokeLine(const PathDataLine&) final;
    void recordStrokeLineWithColorAndThickness(const PathDataLine&, SetInlineStroke&&) final;
    void recordStrokeArc(const PathArc&) final;
    void recordStrokeClosedArc(const PathClosedArc&) final;
    void recordStrokeQuadCurve(const PathDataQuadCurve&) final;
    void recordStrokeBezierCurve(const PathDataBezierCurve&) final;
#endif
    void recordStrokePathSegment(const PathSegment&) final;
    void recordStrokePath(const Path&) final;

    bool recordResourceUse(NativeImage&) final;
    bool recordResourceUse(ImageBuffer&) final;
    bool recordResourceUse(const SourceImage&) final;
    bool recordResourceUse(Font&) final;
    bool recordResourceUse(DecomposedGlyphs&) final;
    bool recordResourceUse(Gradient&) final;
    bool recordResourceUse(Filter&) final;

    void append(Item&& item)
    {
        m_displayList.append(WTFMove(item));
    }

    DisplayList& m_displayList;
};

}
}
