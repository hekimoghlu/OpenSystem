/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

#if USE(SKIA)

#include "GraphicsContext.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkCanvas.h>
#include <skia/effects/SkDashPathEffect.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <wtf/CompletionHandler.h>

namespace WebCore {

class WEBCORE_EXPORT GraphicsContextSkia final : public GraphicsContext {
public:
    GraphicsContextSkia(SkCanvas&, RenderingMode, RenderingPurpose, CompletionHandler<void()>&& = nullptr);
    virtual ~GraphicsContextSkia();

    bool hasPlatformContext() const final;
    SkCanvas* platformContext() const final;

    const DestinationColorSpace& colorSpace() const final;

    void didUpdateState(GraphicsContextState&);

    void setLineCap(LineCap) final;
    void setLineDash(const DashArray&, float) final;
    void setLineJoin(LineJoin) final;
    void setMiterLimit(float) final;

    using GraphicsContext::fillRect;
    void fillRect(const FloatRect&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRect(const FloatRect&, const Color&) final;
    void fillRect(const FloatRect&, Gradient&, const AffineTransform&, RequiresClipToRect = RequiresClipToRect::Yes) final;
    void fillRoundedRectImpl(const FloatRoundedRect&, const Color&) final;
    void fillRectWithRoundedHole(const FloatRect&, const FloatRoundedRect&, const Color&) final;
    void fillPath(const Path&) final;
    void strokeRect(const FloatRect&, float) final;
    void strokePath(const Path&) final;
    void clearRect(const FloatRect&) final;

    void drawNativeImageInternal(NativeImage&, const FloatRect&, const FloatRect&, ImagePaintingOptions) final;
    void drawPattern(NativeImage&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform&, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions) final;
    void drawFilteredImageBuffer(ImageBuffer* sourceImage, const FloatRect& sourceImageRect, Filter&, FilterResults&) final;

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
    void beginTransparencyLayer(CompositeOperator, BlendMode) final;
    void endTransparencyLayer() final;

    void resetClip() final;
    void clip(const FloatRect&) final;
    void clipOut(const FloatRect&) final;
    void clipOut(const Path&) final;
    void clipPath(const Path&, WindRule) final;
    IntRect clipBounds() const final;
    void clipToImageBuffer(ImageBuffer&, const FloatRect&) final;

    RenderingMode renderingMode() const final;

    SkPaint createFillPaint() const;
    SkPaint createStrokePaint() const;

    void drawSkiaText(const sk_sp<SkTextBlob>&, SkScalar, SkScalar, bool, bool);

private:
    bool makeGLContextCurrentIfNeeded() const;

    void setupFillSource(SkPaint&) const;
    void setupStrokeSource(SkPaint&) const;

    enum class ShadowStyle : uint8_t { Outset, Inset };
    sk_sp<SkImageFilter> createDropShadowFilterIfNeeded(ShadowStyle) const;
    bool drawOutsetShadow(SkPaint&, Function<void(const SkPaint&)>&&);

    void drawSkiaRect(const SkRect&, SkPaint&);
    void drawSkiaPath(const SkPath&, SkPaint&);
    void drawSkiaImage(const sk_sp<SkImage>&, const IntSize&, const FloatRect&, const FloatRect&, ImagePaintingOptions);
    void drawSkiaPattern(const sk_sp<SkImage>&, const IntSize&, const FloatRect&, const FloatRect&, const AffineTransform&, const FloatPoint&, const FloatSize&, ImagePaintingOptions);

    class SkiaState {
    public:
        SkiaState() = default;

        struct {
            SkScalar miter { SkFloatToScalar(4) };
            SkPaint::Cap cap { SkPaint::kButt_Cap };
            SkPaint::Join join { SkPaint::kMiter_Join };
            sk_sp<SkPathEffect> dash;
        } m_stroke;
    };

    struct LayerState {
        std::optional<CompositeMode> compositeMode;
    };

    SkCanvas& m_canvas;
    RenderingMode m_renderingMode { RenderingMode::Accelerated };
    RenderingPurpose m_renderingPurpose { RenderingPurpose::Unspecified };
    CompletionHandler<void()> m_destroyNotify;
    SkiaState m_skiaState;
    Vector<SkiaState, 1> m_skiaStateStack;
    Vector<LayerState, 1> m_layerStateStack;
    const DestinationColorSpace m_colorSpace;
};

} // namespace WebCore

#endif // USE(SKIA)
