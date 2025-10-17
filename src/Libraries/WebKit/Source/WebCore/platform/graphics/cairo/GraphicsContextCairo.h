/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#if USE(CAIRO)

#include "GraphicsContext.h"

typedef struct _cairo cairo_t;
typedef struct _cairo_surface cairo_surface_t;

namespace WebCore {

class WEBCORE_EXPORT GraphicsContextCairo final : public GraphicsContext {
public:
    explicit GraphicsContextCairo(RefPtr<cairo_t>&&);
    explicit GraphicsContextCairo(cairo_surface_t*);

#if PLATFORM(WIN)
    GraphicsContextCairo(HDC, bool hasAlpha = false); // FIXME: To be removed.
    explicit GraphicsContextCairo(GraphicsContextCairo*);
#endif

    virtual ~GraphicsContextCairo();

    bool hasPlatformContext() const final;
    GraphicsContextCairo* platformContext() const final;

    void didUpdateState(GraphicsContextState&);

    void setLineCap(LineCap) final;
    void setLineDash(const DashArray&, float) final;
    void setLineJoin(LineJoin) final;
    void setMiterLimit(float) final;

    using GraphicsContext::fillRect;
    void fillRect(const FloatRect&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRect(const FloatRect&, Gradient&, const AffineTransform&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRect(const FloatRect&, const Color&) final;
    void fillRoundedRectImpl(const FloatRoundedRect&, const Color&) final;
    void fillRectWithRoundedHole(const FloatRect&, const FloatRoundedRect&, const Color&) final;
    void fillPath(const Path&) final;
    void strokeRect(const FloatRect&, float) final;
    void strokePath(const Path&) final;
    void clearRect(const FloatRect&) final;

    void drawNativeImageInternal(NativeImage&, const FloatRect&, const FloatRect&, ImagePaintingOptions) final;
    void drawPattern(NativeImage&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform&, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions) final;

    void drawRect(const FloatRect&, float) final;
    void drawLine(const FloatPoint&, const FloatPoint&) final;
    void drawLinesForText(const FloatPoint&, float thickness, const DashArray&, bool, bool, StrokeStyle) final;
    void drawDotsForDocumentMarker(const FloatRect&, DocumentMarkerLineStyle) final;
    void drawEllipse(const FloatRect&) final;

    void drawFocusRing(const Path&, float outlineWidth, const Color&) final;
    void drawFocusRing(const Vector<FloatRect>&, float outlineOffset, float outlineWidth, const Color&) final;

    void save(GraphicsContextState::Purpose = GraphicsContextState::Purpose::SaveRestore) final;
    void restore(GraphicsContextState::Purpose = GraphicsContextState::Purpose::SaveRestore) final;

    void translate(float, float) final;
    void rotate(float) final;
    using GraphicsContext::scale;
    void scale(const FloatSize&) final;
    void concatCTM(const AffineTransform&) final;
    void setCTM(const AffineTransform&) final;
    AffineTransform getCTM(GraphicsContext::IncludeDeviceScale) const final;

    void beginTransparencyLayer(float) final;
    void endTransparencyLayer() final;

    void resetClip() final;
    void clip(const FloatRect&) final;
    void clipOut(const FloatRect&) final;
    void clipOut(const Path&) final;
    void clipPath(const Path&, WindRule) final;
    IntRect clipBounds() const final;
    void clipToImageBuffer(ImageBuffer&, const FloatRect&) final;
    
    RenderingMode renderingMode() const final;

    cairo_t* cr() const;
    Vector<float>& layers();
    void pushImageMask(cairo_surface_t*, const FloatRect&);

    // Exposed as public because freestanding functions use this.
    using GraphicsContext::nativeImageForDrawing;
private:
    RefPtr<cairo_t> m_cr;

    class CairoState;
    CairoState* m_cairoState;
    Vector<CairoState> m_cairoStateStack;

    // Transparency layers.
    Vector<float> m_layers;
};

} // namespace WebCore

#endif // USE(CAIRO)
