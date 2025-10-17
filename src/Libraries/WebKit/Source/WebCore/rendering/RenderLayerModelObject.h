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

#include "PaintPhase.h"
#include "RenderElement.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class BlendingKeyframes;
class LegacyRenderSVGResourceClipper;
class RenderLayer;
class RenderSVGResourceClipper;
class RenderSVGResourceFilter;
class RenderSVGResourceMarker;
class RenderSVGResourceMasker;
class RenderSVGResourcePaintServer;
class SVGGraphicsElement;

class RenderLayerModelObject : public RenderElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderLayerModelObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderLayerModelObject);
public:
    virtual ~RenderLayerModelObject();

    void destroyLayer();

    bool hasSelfPaintingLayer() const;
    RenderLayer* layer() const { return m_layer.get(); }
    CheckedPtr<RenderLayer> checkedLayer() const;

    void styleWillChange(StyleDifference, const RenderStyle& newStyle) override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    virtual bool requiresLayer() const = 0;

    // Returns true if the background is painted opaque in the given rect.
    // The query rect is given in local coordinate system.
    virtual bool backgroundIsKnownToBeOpaqueInRect(const LayoutRect&) const { return false; }

    // Returns false if the rect has no intersection with the applied clip rect. When the context specifies edge-inclusive
    // intersection, this return value allows distinguishing between no intersection and zero-area intersection.
    virtual bool applyCachedClipAndScrollPosition(RepaintRects&, const RenderLayerModelObject*, VisibleRectContext) const { return false; }

    virtual bool isScrollableOrRubberbandableBox() const { return false; }

    bool shouldPlaceVerticalScrollbarOnLeft() const;

    std::optional<LayoutRect> cachedLayerClippedOverflowRect() const;

    bool startAnimation(double timeOffset, const Animation&, const BlendingKeyframes&) override;
    void animationPaused(double timeOffset, const String& name) override;
    void animationFinished(const String& name) override;
    void transformRelatedPropertyDidChange() override;

    void suspendAnimations(MonotonicTime = MonotonicTime()) override;

    // Single source of truth deciding if a SVG renderer should be painted. All SVG renderers
    // use this method to test if they should continue processing in the paint() function or stop.
    bool shouldPaintSVGRenderer(const PaintInfo&, const OptionSet<PaintPhase> relevantPaintPhases = OptionSet<PaintPhase>()) const;

    // Provides the SVG implementation for computeVisibleRectsInContainer().
    // This lives in RenderLayerModelObject, which is the common base-class for all SVG renderers.
    std::optional<RepaintRects> computeVisibleRectsInSVGContainer(const RepaintRects&, const RenderLayerModelObject* container, VisibleRectContext) const;

    // Provides the SVG implementation for mapLocalToContainer().
    // This lives in RenderLayerModelObject, which is the common base-class for all SVG renderers.
    void mapLocalToSVGContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const;

    void applySVGTransform(TransformationMatrix&, const SVGGraphicsElement&, const RenderStyle&, const FloatRect& boundingBox, const std::optional<AffineTransform>& preApplySVGTransformMatrix, const std::optional<AffineTransform>& postApplySVGTransformMatrix, OptionSet<RenderStyle::TransformOperationOption>) const;
    void updateHasSVGTransformFlags();
    virtual bool needsHasSVGTransformFlags() const { ASSERT_NOT_REACHED(); return false; }

    void repaintOrRelayoutAfterSVGTransformChange();

    LayoutPoint nominalSVGLayoutLocation() const { return flooredLayoutPoint(objectBoundingBoxWithoutTransformations().minXMinYCorner()); }
    virtual LayoutPoint currentSVGLayoutLocation() const { ASSERT_NOT_REACHED(); return { }; }
    virtual void setCurrentSVGLayoutLocation(const LayoutPoint&) { ASSERT_NOT_REACHED(); }

    RenderSVGResourcePaintServer* svgFillPaintServerResourceFromStyle(const RenderStyle&) const;
    RenderSVGResourcePaintServer* svgStrokePaintServerResourceFromStyle(const RenderStyle&) const;

    RenderSVGResourceClipper* svgClipperResourceFromStyle() const;
    RenderSVGResourceFilter* svgFilterResourceFromStyle() const;
    RenderSVGResourceMasker* svgMaskerResourceFromStyle() const;
    RenderSVGResourceMarker* svgMarkerStartResourceFromStyle() const;
    RenderSVGResourceMarker* svgMarkerMidResourceFromStyle() const;
    RenderSVGResourceMarker* svgMarkerEndResourceFromStyle() const;

    LegacyRenderSVGResourceClipper* legacySVGClipperResourceFromStyle() const;

    bool pointInSVGClippingArea(const FloatPoint&) const;

    void paintSVGClippingMask(PaintInfo&, const FloatRect& objectBoundingBox) const;
    void paintSVGMask(PaintInfo&, const LayoutPoint& adjustedPaintOffset) const;

    TransformationMatrix* layerTransform() const;

    virtual void updateLayerTransform();
    virtual void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const = 0;
    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox) const;

protected:
    RenderLayerModelObject(Type, Element&, RenderStyle&&, OptionSet<TypeFlag>, TypeSpecificFlags);
    RenderLayerModelObject(Type, Document&, RenderStyle&&, OptionSet<TypeFlag>, TypeSpecificFlags);

    void createLayer();
    void willBeDestroyed() override;

    virtual void updateFromStyle() { }

private:
    RenderSVGResourceMarker* svgMarkerResourceFromStyle(const String& markerResource) const;

    std::unique_ptr<RenderLayer> m_layer;

    // Used to store state between styleWillChange and styleDidChange
    static bool s_wasFloating;
    static bool s_hadLayer;
    static bool s_wasTransformed;
    static bool s_layerWasSelfPainting;
};

// Pixel-snapping (== 'device pixel alignment') helpers.
bool rendererNeedsPixelSnapping(const RenderLayerModelObject&);
FloatRect snapRectToDevicePixelsIfNeeded(const LayoutRect&, const RenderLayerModelObject&);
FloatRect snapRectToDevicePixelsIfNeeded(const FloatRect&, const RenderLayerModelObject&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderLayerModelObject, isRenderLayerModelObject())
