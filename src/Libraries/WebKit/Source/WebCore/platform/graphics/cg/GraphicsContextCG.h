/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#if USE(CG)

#include "ColorSpaceCG.h"
#include "GraphicsContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WEBCORE_EXPORT GraphicsContextCG : public GraphicsContext {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(GraphicsContextCG, WEBCORE_EXPORT);
public:
    enum CGContextSource {
        Unknown,
        CGContextFromCALayer
    };
    GraphicsContextCG(CGContextRef, CGContextSource = CGContextSource::Unknown, std::optional<RenderingMode> knownRenderingMode = std::nullopt);

    ~GraphicsContextCG();

    bool hasPlatformContext() const final;

    // Returns the platform context for any purpose, including draws.
    CGContextRef platformContext() const final;

    const DestinationColorSpace& colorSpace() const final;

    void save(GraphicsContextState::Purpose = GraphicsContextState::Purpose::SaveRestore) final;
    void restore(GraphicsContextState::Purpose = GraphicsContextState::Purpose::SaveRestore) final;

    void drawRect(const FloatRect&, float borderThickness = 1) final;
    void drawLine(const FloatPoint&, const FloatPoint&) final;
    void drawEllipse(const FloatRect&) final;

    void applyStrokePattern() final;
    void applyFillPattern() final;
    void drawPath(const Path&) final;
    void fillPath(const Path&) final;
    void strokePath(const Path&) final;

    void beginTransparencyLayer(float opacity) final;
    void beginTransparencyLayer(CompositeOperator, BlendMode) final;
    void endTransparencyLayer() final;

    void applyDeviceScaleFactor(float factor) final;

    using GraphicsContext::fillRect;
    void fillRect(const FloatRect&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRect(const FloatRect&, const Color&) final;
    void fillRect(const FloatRect&, Gradient&, const AffineTransform&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRoundedRectImpl(const FloatRoundedRect&, const Color&) final;
    void fillRectWithRoundedHole(const FloatRect&, const FloatRoundedRect& roundedHoleRect, const Color&) final;
    void clearRect(const FloatRect&) final;
    void strokeRect(const FloatRect&, float lineWidth) final;

    void fillEllipse(const FloatRect& ellipse) final;
    void strokeEllipse(const FloatRect& ellipse) final;

    bool isCALayerContext() const final;

    RenderingMode renderingMode() const final;

    void resetClip() final;
    void clip(const FloatRect&) final;
    void clipOut(const FloatRect&) final;

    void clipOut(const Path&) final;

    void clipPath(const Path&, WindRule = WindRule::EvenOdd) final;

    void clipToImageBuffer(ImageBuffer&, const FloatRect&) final;

    IntRect clipBounds() const final;

    void setLineCap(LineCap) final;
    void setLineDash(const DashArray&, float dashOffset) final;
    void setLineJoin(LineJoin) final;
    void setMiterLimit(float) final;

    void drawPattern(NativeImage&, const FloatRect& destRect, const FloatRect& tileRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;

    using GraphicsContext::scale;
    void scale(const FloatSize&) final;
    void rotate(float angleInRadians) final;
    void translate(float x, float y) final;

    void concatCTM(const AffineTransform&) final;
    void setCTM(const AffineTransform&) override;

    AffineTransform getCTM(IncludeDeviceScale = PossiblyIncludeDeviceScale) const override;

    void drawFocusRing(const Path&, float outlineWidth, const Color&) final;
    void drawFocusRing(const Vector<FloatRect>&, float outlineOffset, float outlineWidth, const Color&) final;

    void drawLinesForText(const FloatPoint&, float thickness, const DashArray& widths, bool printing, bool doubleLines, StrokeStyle) final;

    void drawDotsForDocumentMarker(const FloatRect&, DocumentMarkerLineStyle) final;

    void beginPage(const IntSize& pageSize) final;
    void endPage() final;

    void setURLForRect(const URL&, const FloatRect&) final;

    void setDestinationForRect(const String& name, const FloatRect&) final;
    void addDestinationAtPoint(const String& name, const FloatPoint&) final;

    bool supportsInternalLinks() const final;

    void didUpdateState(GraphicsContextState&) final;

    virtual bool canUseShadowBlur() const;

    FloatRect roundToDevicePixels(const FloatRect&) const;

    // Returns the platform context for draws.
    CGContextRef contextForDraw();

    // Returns false if there has not been any potential draws since last call.
    // Returns true if there has been potential draws since last call.
    bool consumeHasDrawn();

protected:
    void setCGShadow(const std::optional<GraphicsDropShadow>&, bool shadowsIgnoreTransforms);
    void setCGStyle(const std::optional<GraphicsStyle>&, bool shadowsIgnoreTransforms);

private:
    void drawNativeImageInternal(NativeImage&, const FloatRect& destRect, const FloatRect& srcRect, ImagePaintingOptions = { }) final;

    void clearCGShadow();
    // Returns the platform context for purposes of context state change, not draws.
    CGContextRef contextForState() const;

    const RetainPtr<CGContextRef> m_cgContext;
    mutable std::optional<DestinationColorSpace> m_colorSpace;
    const RenderingMode m_renderingMode : 2; // NOLINT
    const bool m_isLayerCGContext : 1;
    mutable bool m_userToDeviceTransformKnownToBeIdentity : 1 { false };
    // Flag for pending draws. Start with true because we do not know what commands have been scheduled to the context.
    bool m_hasDrawn : 1 { true };
};

CGAffineTransform getUserToBaseCTM(CGContextRef);

} // namespace WebCore

#include "CGContextStateSaver.h"

#endif // USE(CG)
