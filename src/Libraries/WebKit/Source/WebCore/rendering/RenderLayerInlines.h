/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

#include "RenderElementInlines.h"
#include "RenderLayer.h"
#include "RenderObjectInlines.h"
#include "RenderSVGResourceClipper.h"
#include "RenderView.h"
#include "SVGGraphicsElement.h"

namespace WebCore {

inline bool RenderLayer::canPaintTransparencyWithSetOpacity() const { return isBitmapOnly() && !hasNonOpacityTransparency(); }
inline bool RenderLayer::hasBackdropFilter() const { return renderer().hasBackdropFilter(); }
inline bool RenderLayer::hasFilter() const { return renderer().hasFilter(); }
inline bool RenderLayer::hasPerspective() const { return renderer().style().hasPerspective(); }
inline bool RenderLayer::isTransformed() const { return renderer().isTransformed(); }
inline bool RenderLayer::isTransparent() const { return renderer().isTransparent() || renderer().hasMask(); }
inline bool RenderLayer::overlapBoundsIncludeChildren() const { return hasFilter() && renderer().style().filter().hasFilterThatMovesPixels(); }
inline bool RenderLayer::preserves3D() const { return renderer().style().preserves3D(); }
inline int RenderLayer::zIndex() const { return renderer().style().usedZIndex(); }

#if HAVE(CORE_MATERIAL)
inline bool RenderLayer::hasAppleVisualEffect() const { return renderer().hasAppleVisualEffect(); }
inline bool RenderLayer::hasAppleVisualEffectRequiringBackdropFilter() const { return renderer().hasAppleVisualEffectRequiringBackdropFilter(); }
#endif

inline bool RenderLayer::hasBlendMode() const { return renderer().hasBlendMode(); } // FIXME: Why ask the renderer this given we have m_blendMode?

inline bool RenderLayer::canUseOffsetFromAncestor() const
{
    // FIXME: This really needs to know if there are transforms on this layer and any of the layers between it and the ancestor in question.
    return !isTransformed() && !renderer().isRenderOrLegacyRenderSVGRoot();
}

inline bool RenderLayer::paintsWithTransparency(OptionSet<PaintBehavior> paintBehavior) const
{
    if (!renderer().isTransparent() && !hasNonOpacityTransparency())
        return false;
    return (paintBehavior & PaintBehavior::FlattenCompositingLayers) || !isComposited();
}

inline bool RenderLayer::hasNonOpacityTransparency() const
{
    if (renderer().hasMask())
        return true;

    if (hasBlendMode() || isolatesBlending())
        return true;

    if (!renderer().document().settings().layerBasedSVGEngineEnabled())
        return false;

    // SVG clip-paths may use clipping masks, if so, flag this layer as transparent.
    if (auto* svgClipper = renderer().svgClipperResourceFromStyle(); svgClipper && !svgClipper->shouldApplyPathClipping())
        return true;

    return false;
}

inline RenderSVGHiddenContainer* RenderLayer::enclosingSVGHiddenOrResourceContainer() const
{
    return m_enclosingSVGHiddenOrResourceContainer.get();
}

inline const LayoutPoint& RenderLayer::location() const
{
    ASSERT(!renderer().view().frameView().layerAccessPrevented());
    return m_topLeft;
}

inline const IntSize& RenderLayer::size() const
{
    ASSERT(!renderer().view().frameView().layerAccessPrevented());
    return m_layerSize;
}

inline LayoutRect RenderLayer::rect() const
{
    return LayoutRect(location(), size());
}

} // namespace WebCore
