/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#include "ReferencedSVGResources.h"

#include "FilterOperations.h"
#include "LegacyRenderSVGResourceClipper.h"
#include "PathOperation.h"
#include "RenderLayer.h"
#include "RenderSVGPath.h"
#include "RenderStyle.h"
#include "SVGClipPathElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFilterElement.h"
#include "SVGMarkerElement.h"
#include "SVGMaskElement.h"
#include "SVGRenderStyle.h"
#include "SVGResourceElementClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class CSSSVGResourceElementClient final : public SVGResourceElementClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSSVGResourceElementClient);
public:
    CSSSVGResourceElementClient(RenderElement& clientRenderer)
        : m_clientRenderer(clientRenderer)
    {
    }

    void resourceChanged(SVGElement&) final;

    const RenderElement& renderer() const final { return m_clientRenderer.get(); }

private:
    CheckedRef<RenderElement> m_clientRenderer;
};

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSSVGResourceElementClient);

void CSSSVGResourceElementClient::resourceChanged(SVGElement& element)
{
    if (m_clientRenderer->renderTreeBeingDestroyed())
        return;

    if (!m_clientRenderer->document().settings().layerBasedSVGEngineEnabled()) {
        m_clientRenderer->repaint();
        return;
    }

    // Special case for markers. Markers can be attached to RenderSVGPath object. Marker positions are computed
    // once during layout, or if the shape itself changes. Here we manually update the marker positions without
    // requiring a relayout. Instead we can simply repaint the path - via the updateLayerPosition() logic, properly
    // repainting the old repaint boundaries and the new ones (after the marker change).
    if (auto* pathClientRenderer = dynamicDowncast<RenderSVGPath>(m_clientRenderer.get()); pathClientRenderer && is<SVGMarkerElement>(element))
        pathClientRenderer->updateMarkerPositions();

    m_clientRenderer->repaintOldAndNewPositionsForSVGRenderer();
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ReferencedSVGResources);

ReferencedSVGResources::ReferencedSVGResources(RenderElement& renderer)
    : m_renderer(renderer)
{
}

ReferencedSVGResources::~ReferencedSVGResources()
{
    Ref treeScope = m_renderer->treeScopeForSVGReferences();
    for (auto& targetID : copyToVector(m_elementClients.keys()))
        removeClientForTarget(treeScope, targetID);
}

void ReferencedSVGResources::addClientForTarget(SVGElement& targetElement, const AtomString& targetID)
{
    m_elementClients.ensure(targetID, [&] {
        auto client = makeUnique<CSSSVGResourceElementClient>(m_renderer);
        targetElement.addReferencingCSSClient(*client);
        return client;
    });
}

void ReferencedSVGResources::removeClientForTarget(TreeScope& treeScope, const AtomString& targetID)
{
    auto client = m_elementClients.take(targetID);

    if (RefPtr targetElement = dynamicDowncast<SVGElement>(treeScope.getElementById(targetID)))
        targetElement->removeReferencingCSSClient(*client);
}

ReferencedSVGResources::SVGElementIdentifierAndTagPairs ReferencedSVGResources::referencedSVGResourceIDs(const RenderStyle& style, const Document& document)
{
    SVGElementIdentifierAndTagPairs referencedResources;
    if (auto* clipPath = dynamicDowncast<ReferencePathOperation>(style.clipPath())) {
        if (!clipPath->fragment().isEmpty())
            referencedResources.append({ clipPath->fragment(), { SVGNames::clipPathTag } });
    }

    if (style.hasFilter()) {
        const auto& filterOperations = style.filter();
        for (auto& operation : filterOperations) {
            if (RefPtr referenceFilterOperation = dynamicDowncast<ReferenceFilterOperation>(operation)) {
                if (!referenceFilterOperation->fragment().isEmpty())
                    referencedResources.append({ referenceFilterOperation->fragment(), { SVGNames::filterTag } });
            }
        }
    }

    if (!document.settings().layerBasedSVGEngineEnabled())
        return referencedResources;

    if (style.hasPositionedMask()) {
        // FIXME: We should support all the values in the CSS mask property, but for now just use the first mask-image if it's a reference.
        auto* maskImage = style.maskImage();
        auto reresolvedURL = maskImage ? maskImage->reresolvedURL(document) : URL();

        if (!reresolvedURL.isEmpty()) {
            auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(reresolvedURL.string(), document);
            if (!resourceID.isEmpty())
                referencedResources.append({ resourceID, { SVGNames::maskTag } });
        }
    }

    const auto& svgStyle = style.svgStyle();
    if (svgStyle.hasMarkers()) {
        if (auto markerStartResource = svgStyle.markerStartResource(); !markerStartResource.isEmpty()) {
            auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(markerStartResource, document);
            if (!resourceID.isEmpty())
                referencedResources.append({ resourceID, { SVGNames::markerTag } });
        }

        if (auto markerMidResource = svgStyle.markerMidResource(); !markerMidResource.isEmpty()) {
            auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(markerMidResource, document);
            if (!resourceID.isEmpty())
                referencedResources.append({ resourceID, { SVGNames::markerTag } });
        }

        if (auto markerEndResource = svgStyle.markerEndResource(); !markerEndResource.isEmpty()) {
            auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(markerEndResource, document);
            if (!resourceID.isEmpty())
                referencedResources.append({ resourceID, { SVGNames::markerTag } });
        }
    }

    if (svgStyle.fillPaintType() >= SVGPaintType::URINone) {
        auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(svgStyle.fillPaintUri(), document);
        if (!resourceID.isEmpty())
            referencedResources.append({ resourceID, { SVGNames::linearGradientTag, SVGNames::radialGradientTag, SVGNames::patternTag } });
    }

    if (svgStyle.strokePaintType() >= SVGPaintType::URINone) {
        auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(svgStyle.strokePaintUri(), document);
        if (!resourceID.isEmpty())
            referencedResources.append({ resourceID, { SVGNames::linearGradientTag, SVGNames::radialGradientTag, SVGNames::patternTag } });
    }

    return referencedResources;
}

void ReferencedSVGResources::updateReferencedResources(TreeScope& treeScope, const ReferencedSVGResources::SVGElementIdentifierAndTagPairs& referencedResources)
{
    UncheckedKeyHashSet<AtomString> oldKeys;
    for (auto& key : m_elementClients.keys())
        oldKeys.add(key);

    for (auto& [targetID, tagNames] : referencedResources) {
        RefPtr element = elementForResourceIDs(treeScope, targetID, tagNames);
        if (!element)
            continue;

        addClientForTarget(*element, targetID);
        oldKeys.remove(targetID);
    }

    for (auto& targetID : oldKeys)
        removeClientForTarget(treeScope, targetID);
}

// SVG code uses getRenderSVGResourceById<>, but that works in terms of renderers. We need to find resources
// before the render tree is fully constructed, so this works on Elements.
RefPtr<SVGElement> ReferencedSVGResources::elementForResourceID(TreeScope& treeScope, const AtomString& resourceID, const SVGQualifiedName& tagName)
{
    RefPtr element = dynamicDowncast<SVGElement>(treeScope.getElementById(resourceID));
    if (!element || !element->hasTagName(tagName))
        return nullptr;

    return element;
}

RefPtr<SVGElement> ReferencedSVGResources::elementForResourceIDs(TreeScope& treeScope, const AtomString& resourceID, const SVGQualifiedNames& tagNames)
{
    RefPtr element = dynamicDowncast<SVGElement>(treeScope.getElementById(resourceID));
    if (!element)
        return nullptr;

    for (const auto& tagName : tagNames) {
        if (element->hasTagName(tagName))
            return element;
    }

    return nullptr;
}

RefPtr<SVGClipPathElement> ReferencedSVGResources::referencedClipPathElement(TreeScope& treeScope, const ReferencePathOperation& clipPath)
{
    if (clipPath.fragment().isEmpty())
        return nullptr;
    RefPtr element = elementForResourceID(treeScope, clipPath.fragment(), SVGNames::clipPathTag);
    return element ? downcast<SVGClipPathElement>(WTFMove(element)) : nullptr;
}

RefPtr<SVGMarkerElement> ReferencedSVGResources::referencedMarkerElement(TreeScope& treeScope, const String& markerResource)
{
    auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(markerResource, treeScope.protectedDocumentScope());
    if (resourceID.isEmpty())
        return nullptr;

    RefPtr element = elementForResourceID(treeScope, resourceID, SVGNames::markerTag);
    return element ? downcast<SVGMarkerElement>(WTFMove(element)) : nullptr;
}

RefPtr<SVGMaskElement> ReferencedSVGResources::referencedMaskElement(TreeScope& treeScope, const StyleImage& maskImage)
{
    auto reresolvedURL = maskImage.reresolvedURL(treeScope.protectedDocumentScope());
    if (reresolvedURL.isEmpty())
        return nullptr;

    auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(reresolvedURL.string(), treeScope.protectedDocumentScope());
    if (resourceID.isEmpty())
        return nullptr;

    return referencedMaskElement(treeScope, resourceID);
}

RefPtr<SVGMaskElement> ReferencedSVGResources::referencedMaskElement(TreeScope& treeScope, const AtomString& resourceID)
{
    RefPtr element = elementForResourceID(treeScope, resourceID, SVGNames::maskTag);
    return element ? downcast<SVGMaskElement>(WTFMove(element)) : nullptr;
}

RefPtr<SVGElement> ReferencedSVGResources::referencedPaintServerElement(TreeScope& treeScope, const String& uri)
{
    auto resourceID = SVGURIReference::fragmentIdentifierFromIRIString(uri, treeScope.protectedDocumentScope());
    if (resourceID.isEmpty())
        return nullptr;

    return elementForResourceIDs(treeScope, resourceID, { SVGNames::linearGradientTag, SVGNames::radialGradientTag, SVGNames::patternTag });
}

RefPtr<SVGFilterElement> ReferencedSVGResources::referencedFilterElement(TreeScope& treeScope, const ReferenceFilterOperation& referenceFilter)
{
    if (referenceFilter.fragment().isEmpty())
        return nullptr;
    RefPtr element = elementForResourceID(treeScope, referenceFilter.fragment(), SVGNames::filterTag);
    return element ? downcast<SVGFilterElement>(WTFMove(element)) : nullptr;
}

LegacyRenderSVGResourceClipper* ReferencedSVGResources::referencedClipperRenderer(TreeScope& treeScope, const ReferencePathOperation& clipPath)
{
    if (clipPath.fragment().isEmpty())
        return nullptr;
    // For some reason, SVG stores a cache of id -> renderer, rather than just using getElementById() and renderer().
    return getRenderSVGResourceById<LegacyRenderSVGResourceClipper>(treeScope, clipPath.fragment());
}

LegacyRenderSVGResourceContainer* ReferencedSVGResources::referencedRenderResource(TreeScope& treeScope, const AtomString& fragment)
{
    if (fragment.isEmpty())
        return nullptr;
    return getRenderSVGResourceContainerById(treeScope, fragment);
}

} // namespace WebCore
