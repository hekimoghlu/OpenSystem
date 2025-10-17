/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#include "CairoPaintingOperation.h"
#include "GraphicsContext.h"

namespace WebCore {
namespace Cairo {

class OperationRecorder final : public WebCore::GraphicsContext {
public:
    OperationRecorder(PaintingOperations&);

private:
    bool hasPlatformContext() const override { return false; }
    PlatformGraphicsContext* platformContext() const override { return nullptr; }

    void didUpdateState(WebCore::GraphicsContextState&) override;

    void setLineCap(WebCore::LineCap) override;
    void setLineDash(const WebCore::DashArray&, float) override;
    void setLineJoin(WebCore::LineJoin) override;
    void setMiterLimit(float) override;

    void fillRect(const WebCore::FloatRect&, WebCore::GraphicsContext::RequiresClipToRect = WebCore::GraphicsContext::RequiresClipToRect::Yes) override;
    void fillRect(const WebCore::FloatRect&, const WebCore::Color&) override;
    void fillRect(const WebCore::FloatRect&, WebCore::Gradient&) override;
    void fillRect(const WebCore::FloatRect&, WebCore::Gradient&, const WebCore::AffineTransform&, WebCore::GraphicsContext::RequiresClipToRect = WebCore::GraphicsContext::RequiresClipToRect::Yes) override;
    void fillRect(const WebCore::FloatRect&, const WebCore::Color&, WebCore::CompositeOperator, WebCore::BlendMode) override;
    void fillRoundedRectImpl(const WebCore::FloatRoundedRect&, const WebCore::Color&) override { ASSERT_NOT_REACHED(); }
    void fillRoundedRect(const WebCore::FloatRoundedRect&, const WebCore::Color&, WebCore::BlendMode) override;
    void fillRectWithRoundedHole(const WebCore::FloatRect&, const WebCore::FloatRoundedRect&, const WebCore::Color&) override;
    void fillPath(const WebCore::Path&) override;
    void fillEllipse(const WebCore::FloatRect&) override;
    void strokeRect(const WebCore::FloatRect&, float) override;
    void strokePath(const WebCore::Path&) override;
    void strokeEllipse(const WebCore::FloatRect&) override;
    void clearRect(const WebCore::FloatRect&) override;

    void drawGlyphs(const WebCore::Font&, std::span<const WebCore::GlyphBufferGlyph>, std::span<const WebCore::GlyphBufferAdvance>, const WebCore::FloatPoint&, WebCore::FontSmoothingMode) override;
    void drawDecomposedGlyphs(const WebCore::Font&, const WebCore::DecomposedGlyphs&) override;

    void drawImageBuffer(WebCore::ImageBuffer&, const WebCore::FloatRect& destination, const WebCore::FloatRect& source, WebCore::ImagePaintingOptions) override;
    void drawFilteredImageBuffer(WebCore::ImageBuffer*, const WebCore::FloatRect&, WebCore::Filter&, WebCore::FilterResults&) override;
    void drawNativeImageInternal(WebCore::NativeImage&, const WebCore::FloatRect&, const WebCore::FloatRect&, WebCore::ImagePaintingOptions) override;
    void drawPattern(WebCore::NativeImage&, const WebCore::FloatRect&, const WebCore::FloatRect&, const WebCore::AffineTransform&, const WebCore::FloatPoint&, const WebCore::FloatSize&, WebCore::ImagePaintingOptions) override;

    void drawRect(const WebCore::FloatRect&, float) override;
    void drawLine(const WebCore::FloatPoint&, const WebCore::FloatPoint&) override;
    void drawLinesForText(const WebCore::FloatPoint&, float thickness, const WebCore::DashArray&, bool, bool, WebCore::StrokeStyle) override;
    void drawDotsForDocumentMarker(const WebCore::FloatRect&, WebCore::DocumentMarkerLineStyle) override;
    void drawEllipse(const WebCore::FloatRect&) override;

    void drawFocusRing(const WebCore::Path&, float outlineWidth, const WebCore::Color&) override;
    void drawFocusRing(const Vector<WebCore::FloatRect>&, float outlineOffset, float outlineWidth, const WebCore::Color&) override;

    void save(WebCore::GraphicsContextState::Purpose = WebCore::GraphicsContextState::Purpose::SaveRestore) override;
    void restore(WebCore::GraphicsContextState::Purpose = WebCore::GraphicsContextState::Purpose::SaveRestore) override;

    void translate(float, float) override;
    void rotate(float angleInRadians) override;
    void scale(const WebCore::FloatSize&) override;
    void concatCTM(const WebCore::AffineTransform&) override;
    void setCTM(const WebCore::AffineTransform&) override;
    WebCore::AffineTransform getCTM(WebCore::GraphicsContext::IncludeDeviceScale) const override;

    void beginTransparencyLayer(float) override;
    void beginTransparencyLayer(WebCore::CompositeOperator, WebCore::BlendMode) override;
    void endTransparencyLayer() override;

    void resetClip() override;
    void clip(const WebCore::FloatRect&) override;
    void clipOut(const WebCore::FloatRect&) override;
    void clipOut(const WebCore::Path&) override;
    void clipPath(const WebCore::Path&, WebCore::WindRule) override;
    WebCore::IntRect clipBounds() const override;
    void clipToImageBuffer(WebCore::ImageBuffer&, const WebCore::FloatRect&) override;
#if ENABLE(VIDEO)
    void drawVideoFrame(WebCore::VideoFrame&, const WebCore::FloatRect& destination, WebCore::ImageOrientation, bool shouldDiscardAlpha) override;
#endif

    void applyDeviceScaleFactor(float) override;

    void append(std::unique_ptr<PaintingOperation>&&);
    PaintingOperations& m_commandList;

    struct State {
        WebCore::AffineTransform ctm;
        WebCore::AffineTransform ctmInverse;
        WebCore::FloatRect clipBounds;
    };
    Vector<State, 32> m_stateStack;
};

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
