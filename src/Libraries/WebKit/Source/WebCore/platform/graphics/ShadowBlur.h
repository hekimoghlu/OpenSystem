/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

#include "Color.h"
#include "FloatRect.h"
#include "FloatRoundedRect.h"
#include "GraphicsStyle.h"
#include <wtf/Function.h>
#include <wtf/Noncopyable.h>

namespace WebCore {

class AffineTransform;
class GraphicsContext;
struct DropShadow;
class ImageBuffer;

class ShadowBlur {
    WTF_MAKE_NONCOPYABLE(ShadowBlur);
public:
    enum ShadowType {
        NoShadow,
        SolidShadow,
        BlurShadow
    };

    ShadowBlur();
    ShadowBlur(const FloatSize& radius, const FloatSize& offset, const Color&, bool shadowsIgnoreTransforms = false);
    ShadowBlur(const GraphicsDropShadow&, bool shadowsIgnoreTransforms = false);

    void setShadowValues(const FloatSize&, const FloatSize& , const Color&, bool ignoreTransforms = false);

    void setShadowsIgnoreTransforms(bool ignoreTransforms) { m_shadowsIgnoreTransforms = ignoreTransforms; }
    bool shadowsIgnoreTransforms() const { return m_shadowsIgnoreTransforms; }

    void drawRectShadow(GraphicsContext&, const FloatRoundedRect&);
    void drawInsetShadow(GraphicsContext&, const FloatRect&, const FloatRoundedRect& holeRect);

    using DrawBufferCallback = Function<void(ImageBuffer&, const FloatPoint&, const FloatSize&)>;
    using DrawImageCallback = Function<void(ImageBuffer&, const FloatRect&, const FloatRect&)>;
    using FillRectCallback = Function<void(const FloatRect&, const Color&)>;
    using FillRectWithHoleCallback = Function<void(const FloatRect&, const FloatRect&, const Color&)>;
    using DrawShadowCallback = Function<void(GraphicsContext&)>;

    // DrawBufferCallback is for drawing shadow without tiling.
    // DrawImageCallback and FillRectCallback is for drawing shadow with tiling.
    void drawRectShadow(const AffineTransform&, const IntRect& clipBounds, const FloatRoundedRect& shadowedRect, const DrawBufferCallback&, const DrawImageCallback&, const FillRectCallback&);
    void drawInsetShadow(const AffineTransform&, const IntRect& clipBounds, const FloatRect& fullRect, const FloatRoundedRect& holeRect, const DrawBufferCallback&, const DrawImageCallback&, const FillRectWithHoleCallback&);
    void drawShadowLayer(const AffineTransform&, const IntRect& clipBounds, const FloatRect& layerArea, const DrawShadowCallback&, const DrawBufferCallback&);

    void blurLayerImage(std::span<uint8_t>, const IntSize&, int stride);

    void clear();

    ShadowType type() const { return m_type; }

private:
    void updateShadowBlurValues();

    void drawShadowBuffer(GraphicsContext&, ImageBuffer&, const FloatPoint&, const FloatSize&);

    void adjustBlurRadius(const AffineTransform&);

    enum ShadowDirection {
        OuterShadow,
        InnerShadow
    };

    struct LayerImageProperties {
        FloatSize shadowedResultSize; // Size of the result of shadowing which is same as shadowedRect + blurred edges.
        FloatPoint layerOrigin; // Top-left corner of the (possibly clipped) bounding rect to draw the shadow to.
        FloatSize layerSize; // Size of layerImage pixels that need blurring.
        FloatSize layerContextTranslation; // Translation to apply to layerContext for the shadow to be correctly clipped.
    };

    std::optional<ShadowBlur::LayerImageProperties> calculateLayerBoundingRect(const AffineTransform&, const FloatRect& layerArea, const IntRect& clipRect);
    IntSize templateSize(const IntSize& blurredEdgeSize, const FloatRoundedRect::Radii&) const;

    void blurShadowBuffer(ImageBuffer& layerImage, const IntSize& templateSize);
    void blurAndColorShadowBuffer(ImageBuffer& layerImage, const IntSize& templateSize);

    void drawInsetShadowWithoutTiling(const AffineTransform&, const FloatRect& fullRect, const FloatRoundedRect& holeRect, const LayerImageProperties&, const DrawBufferCallback&);
    void drawInsetShadowWithTiling(const AffineTransform&, const FloatRect& fullRect, const FloatRoundedRect& holeRect, const IntSize& shadowTemplateSize, const IntSize& blurredEdgeSize, const DrawImageCallback&, const FillRectWithHoleCallback&);
    void drawInsetShadowWithTilingWithLayerImageBuffer(ImageBuffer& layerImage, const AffineTransform&, const FloatRect& fullRect, const FloatRoundedRect& holeRect, const IntSize& shadowTemplateSize, const IntSize& blurredEdgeSize, const DrawImageCallback&, const FillRectWithHoleCallback&, const FloatRect& templateBounds, const FloatRect& templateHole, bool redrawNeeded = true);

    void drawRectShadowWithoutTiling(const AffineTransform&, const FloatRoundedRect& shadowedRect, const LayerImageProperties&, const DrawBufferCallback&);
    void drawRectShadowWithTiling(const AffineTransform&, const FloatRoundedRect& shadowedRect, const IntSize& shadowTemplateSize, const IntSize& blurredEdgeSize, const DrawImageCallback&, const FillRectCallback&, const LayerImageProperties&);
    void drawRectShadowWithTilingWithLayerImageBuffer(ImageBuffer& layerImage, const AffineTransform&, const FloatRoundedRect& shadowedRect, const IntSize& templateSize, const IntSize& edgeSize, const DrawImageCallback&, const FillRectCallback&, const FloatRect& templateShadow, bool redrawNeeded = true);

    void drawLayerPiecesAndFillCenter(ImageBuffer& layerImage, const FloatRect& shadowBounds, const FloatRoundedRect::Radii&, const IntSize& roundedRadius, const IntSize& templateSize, const DrawImageCallback&, const FillRectCallback&);
    void drawLayerPieces(ImageBuffer& layerImage, const FloatRect& shadowBounds, const FloatRoundedRect::Radii&, const IntSize& roundedRadius, const IntSize& templateSize, const DrawImageCallback&);

    IntSize blurredEdgeSize() const;

    ShadowType m_type { NoShadow };

    Color m_color;
    FloatSize m_blurRadius;
    FloatSize m_offset;

    bool m_shadowsIgnoreTransforms { false };
};

} // namespace WebCore
