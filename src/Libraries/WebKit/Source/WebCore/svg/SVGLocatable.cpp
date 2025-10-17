/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#include "config.h"
#include "SVGLocatable.h"

#include "RenderElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGraphicsElement.h"
#include "SVGImageElement.h"
#include "SVGLayerTransformComputation.h"
#include "SVGMatrix.h"
#include "SVGNames.h"
#include "TransformState.h"

namespace WebCore {

// FIXME: This doesn't match SVGElement::viewportElement() as it has an extra check for
// foreign object.
static bool isViewportElement(const SVGElement* element)
{
    if (!element)
        return false;

    return element->hasTagName(SVGNames::svgTag)
        || element->hasTagName(SVGNames::symbolTag)
        || element->hasTagName(SVGNames::foreignObjectTag)
        || is<SVGImageElement>(*element);
}

SVGElement* SVGLocatable::nearestViewportElement(const SVGElement* element)
{
    ASSERT(element);
    for (RefPtr current = element->parentOrShadowHostElement(); current; current = current->parentOrShadowHostElement()) {
        auto* svgElement = dynamicDowncast<SVGElement>(*current);
        if (isViewportElement(svgElement))
            return svgElement;
    }

    return nullptr;
}

SVGElement* SVGLocatable::farthestViewportElement(const SVGElement* element)
{
    ASSERT(element);
    SUPPRESS_UNCOUNTED_LOCAL SVGElement* farthest = nullptr;
    for (RefPtr current = element->parentOrShadowHostElement(); current; current = current->parentOrShadowHostElement()) {
        auto* svgElement = dynamicDowncast<SVGElement>(*current);
        if (isViewportElement(svgElement))
            farthest = svgElement;
    }
    return farthest;
}

FloatRect SVGLocatable::getBBox(SVGElement* element, StyleUpdateStrategy styleUpdateStrategy)
{
    ASSERT(element);
    if (styleUpdateStrategy == AllowStyleUpdate)
        element->protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, element);

    // FIXME: Eventually we should support getBBox for detached elements.
    if (!element->renderer())
        return FloatRect();

    return element->renderer()->objectBoundingBox();
}

AffineTransform SVGLocatable::computeCTM(SVGElement* element, CTMScope mode, StyleUpdateStrategy styleUpdateStrategy)
{
    ASSERT(element);
    if (styleUpdateStrategy == AllowStyleUpdate)
        element->protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, element);

    RefPtr stopAtElement = mode == NearestViewportScope ? nearestViewportElement(element) : nullptr;

    if (element->document().settings().layerBasedSVGEngineEnabled()) {
        // Rudimentary support for operations on "detached" elements.
        CheckedPtr renderer = dynamicDowncast<RenderLayerModelObject>(element->renderer());
        if (!renderer)
            return element->localCoordinateSpaceTransform(mode);

        auto trackingMode { mode == SVGLocatable::ScreenScope ? TransformState::TrackSVGScreenCTMMatrix : TransformState::TrackSVGCTMMatrix };
        CheckedPtr stopAtRenderer = dynamicDowncast<RenderLayerModelObject>(stopAtElement ? stopAtElement->renderer() : nullptr);
        return SVGLayerTransformComputation(*renderer).computeAccumulatedTransform(stopAtRenderer.get(), trackingMode);
    }

    AffineTransform ctm;

    for (Element* currentElement = element; currentElement; currentElement = currentElement->parentOrShadowHostElement()) {
        RefPtr svgElement = dynamicDowncast<SVGElement>(*currentElement);
        if (!svgElement)
            break;

        ctm = svgElement->localCoordinateSpaceTransform(mode).multiply(ctm);

        // For getCTM() computation, stop at the nearest viewport element
        if (currentElement == stopAtElement)
            break;
    }

    return ctm;
}

}
