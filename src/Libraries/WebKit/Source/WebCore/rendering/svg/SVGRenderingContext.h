/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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

#include "ImageBuffer.h"
#include "PaintInfo.h"

namespace WebCore {

class AffineTransform;
class FloatRect;
class RenderElement;
class RenderObject;
class LegacyRenderSVGResourceFilter;

class SVGRenderingContext {
public:
    enum NeedsGraphicsContextSave {
        SaveGraphicsContext,
        DontSaveGraphicsContext,
    };

    // Does not start rendering.
    SVGRenderingContext()
    {
    }

    SVGRenderingContext(SVGRenderingContext&&);
    SVGRenderingContext(const SVGRenderingContext&) = delete;

    SVGRenderingContext(RenderElement& object, PaintInfo& paintinfo, NeedsGraphicsContextSave needsGraphicsContextSave = DontSaveGraphicsContext)
    {
        prepareToRenderSVGContent(object, paintinfo, needsGraphicsContextSave);
    }

    // Automatically finishes context rendering.
    ~SVGRenderingContext();

    // Used by all SVG renderers who apply clip/filter/etc. resources to the renderer content.
    void prepareToRenderSVGContent(RenderElement&, PaintInfo&, NeedsGraphicsContextSave = DontSaveGraphicsContext);
    bool isRenderingPrepared() const { return m_renderingFlags & RenderingPrepared; }

    bool pathClippingIsEntirelyWithinRendererContents() const { return m_pathClippingIsEntirelyWithinRendererContents; }

    static void renderSubtreeToContext(GraphicsContext&, RenderElement&, const AffineTransform&);
    static void clipToImageBuffer(GraphicsContext&, const FloatRect& targetRect, const FloatSize& scale, RefPtr<ImageBuffer>&, bool safeToClear);

    static float calculateScreenFontSizeScalingFactor(const RenderObject&);
    static AffineTransform calculateTransformationToOutermostCoordinateSystem(const RenderObject&);

    static IntRect calculateImageBufferRect(const FloatRect& targetRect, const AffineTransform& absoluteTransform)
    {
        return enclosingIntRect(absoluteTransform.mapRect(targetRect));
    }

    // Support for the buffered-rendering hint.
    bool bufferForeground(RefPtr<ImageBuffer>&);

    const RenderElement* renderer() const { return m_renderer; }

private:
    // To properly revert partially successful initializtions in the destructor, we record all successful steps.
    enum RenderingFlags {
        RenderingPrepared = 1,
        RestoreGraphicsContext = 1 << 1,
        EndOpacityLayer = 1 << 2,
        EndFilterLayer = 1 << 3,
        PrepareToRenderSVGContentWasCalled = 1 << 4
    };

    // List of those flags which require actions during the destructor.
    static constexpr int ActionsNeeded = RestoreGraphicsContext | EndOpacityLayer | EndFilterLayer;

    RenderElement* m_renderer { nullptr };
    PaintInfo* m_paintInfo { nullptr };
    GraphicsContext* m_savedContext  { nullptr };
    LegacyRenderSVGResourceFilter* m_filter  { nullptr };
    LayoutRect m_savedPaintRect;
    int m_renderingFlags { 0 };
    // True with path-based clipping is known to contrain the clipped area to within the renderer; used to optimize away a context clip.
    bool m_pathClippingIsEntirelyWithinRendererContents { false };
};

} // namespace WebCore
