/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "SVGResourcesCache.h"

#include "ElementInlines.h"
#include "LegacyRenderSVGResourceContainer.h"
#include "SVGRenderStyle.h"
#include "SVGResources.h"
#include "SVGResourcesCycleSolver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGResourcesCache);

SVGResourcesCache::SVGResourcesCache() = default;

SVGResourcesCache::~SVGResourcesCache() = default;

void SVGResourcesCache::addResourcesFromRenderer(RenderElement& renderer, const RenderStyle& style)
{
    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    ASSERT(!renderer.hasCachedSVGResource());
    ASSERT(!m_cache.contains(renderer));

    // Build a list of all resources associated with the passed RenderObject
    auto newResources = SVGResources::buildCachedResources(renderer, style);
    if (!newResources)
        return;

    // Put object in cache.
    SVGResources& resources = *m_cache.add(renderer, WTFMove(newResources)).iterator->value;
    renderer.setHasCachedSVGResource(true);

    // Run cycle-detection _afterwards_, so self-references can be caught as well.
    SVGResourcesCycleSolver::resolveCycles(renderer, resources);

    // Walk resources and register the render object at each resources.
    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> resourceSet;
    resources.buildSetOfResources(resourceSet);

    for (auto& resourceContainer : resourceSet)
        resourceContainer.addClient(renderer);
}

void SVGResourcesCache::removeResourcesFromRenderer(RenderElement& renderer)
{
    if (!renderer.hasCachedSVGResource())
        return;

    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    auto resources = m_cache.take(renderer);
    renderer.setHasCachedSVGResource(false);
    if (!resources)
        return;

    // Walk resources and register the render object at each resources.
    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> resourceSet;
    resources->buildSetOfResources(resourceSet);

    for (auto& resourceContainer : resourceSet)
        resourceContainer.removeClient(renderer);
}

static inline SVGResourcesCache& resourcesCacheFromRenderer(const RenderElement& renderer)
{
    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    return renderer.document().svgExtensions().resourcesCache();
}

SVGResources* SVGResourcesCache::cachedResourcesForRenderer(const RenderElement& renderer)
{
    if (!renderer.hasCachedSVGResource())
        return nullptr;
    return resourcesCacheFromRenderer(renderer).m_cache.get(renderer);
}

static bool hasPaintResourceRequiringRemovalOnClientLayoutChange(LegacyRenderSVGResource* resource)
{
    return resource && resource->resourceType() == PatternResourceType;
}

static bool hasResourcesRequiringRemovalOnClientLayoutChange(SVGResources& resources)
{
    return resources.masker() || resources.filter() || hasPaintResourceRequiringRemovalOnClientLayoutChange(resources.fill()) || hasPaintResourceRequiringRemovalOnClientLayoutChange(resources.stroke());
}

void SVGResourcesCache::clientLayoutChanged(RenderElement& renderer)
{
    if (!renderer.hasCachedSVGResource())
        return;

    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    auto* resources = SVGResourcesCache::cachedResourcesForRenderer(renderer);
    if (!resources)
        return;

    // Invalidate the resources if either the RenderElement itself changed,
    // or we have filter resources, which could depend on the layout of children.
    if ((renderer.selfNeedsLayout() || resources->filter()) && hasResourcesRequiringRemovalOnClientLayoutChange(*resources))
        resources->removeClientFromCacheAndMarkForInvalidation(renderer, false);
}

static inline bool rendererCanHaveResources(RenderObject& renderer)
{
    return renderer.node() && renderer.node()->isSVGElement() && !renderer.isRenderSVGInlineText();
}

void SVGResourcesCache::clientStyleChanged(RenderElement& renderer, StyleDifference diff, const RenderStyle* oldStyle, const RenderStyle& newStyle)
{
    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    ASSERT(!renderer.element() || renderer.element()->isSVGElement());

    if (!renderer.parent())
        return;

    // For filter primitives, when diff is Repaint or RepaintIsText, the
    // SVGFE*Element will decide whether the modified CSS properties require a
    // relayout or repaint.
    //
    // Since diff can be Equal even if we have have a filter property change
    // (due to how RenderElement::adjustStyleDifference works), in general we
    // want to continue to the comparison of oldStyle and newStyle below, and
    // so we don't return early just when diff == StyleDifference::Equal. But
    // this isn't necessary for filter primitives, to which the filter property
    // doesn't apply, so we check for it here too.
    if (renderer.isLegacyRenderSVGResourceFilterPrimitive() && (diff == StyleDifference::Equal || diff == StyleDifference::Repaint || diff == StyleDifference::RepaintIfText))
        return;

    auto hasStyleDifferencesAffectingResources = [&] {
        if (!rendererCanHaveResources(renderer))
            return false;

        if (!oldStyle)
            return true;

        if (!arePointingToEqualData(oldStyle->clipPath(), newStyle.clipPath()))
            return true;

        // RenderSVGResourceMarker only supports SVG <mask> references.
        if (!arePointingToEqualData(oldStyle->maskImage(), newStyle.maskImage()))
            return true;

        if (oldStyle->filter() != newStyle.filter())
            return true;

        // -apple-color-filter affects gradients.
        if (oldStyle->appleColorFilter() != newStyle.appleColorFilter())
            return true;

        Ref oldSVGStyle = oldStyle->svgStyle();
        Ref newSVGStyle = newStyle.svgStyle();

        if (oldSVGStyle->fillPaintUri() != newSVGStyle->fillPaintUri())
            return true;

        if (oldSVGStyle->strokePaintUri() != newSVGStyle->strokePaintUri())
            return true;

        return false;
    };

    if (hasStyleDifferencesAffectingResources()) {
        auto& cache = resourcesCacheFromRenderer(renderer);
        cache.removeResourcesFromRenderer(renderer);
        cache.addResourcesFromRenderer(renderer, newStyle);
    }

    LegacyRenderSVGResource::markForLayoutAndParentResourceInvalidation(renderer, false);
}

void SVGResourcesCache::clientWasAddedToTree(RenderObject& renderer)
{
    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    if (renderer.isAnonymous())
        return;

    LegacyRenderSVGResource::markForLayoutAndParentResourceInvalidation(renderer, false);

    if (!rendererCanHaveResources(renderer))
        return;
    RenderElement& elementRenderer = downcast<RenderElement>(renderer);
    resourcesCacheFromRenderer(elementRenderer).addResourcesFromRenderer(elementRenderer, elementRenderer.style());
}

void SVGResourcesCache::clientWillBeRemovedFromTree(RenderObject& renderer)
{
    if (!rendererCanHaveResources(renderer))
        return;
    RenderElement& elementRenderer = downcast<RenderElement>(renderer);

    if (!elementRenderer.hasCachedSVGResource())
        return;

    // While LBSE does not make use of SVGResourcesCache, we might get here after switching from legacy to LBSE
    // and destructing the legacy tree -- when LBSE is already activated - don't assert here that this is not reached.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        return;

    if (renderer.isAnonymous())
        return;

    LegacyRenderSVGResource::markForLayoutAndParentResourceInvalidation(renderer, false);
    resourcesCacheFromRenderer(elementRenderer).removeResourcesFromRenderer(elementRenderer);
}

void SVGResourcesCache::clientDestroyed(RenderElement& renderer)
{
    if (!renderer.hasCachedSVGResource())
        return;

    // While LBSE does not make use of SVGResourcesCache, we might get here after switching from legacy to LBSE
    // and destructing the legacy tree -- when LBSE is already activated - don't assert here that this is not reached.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        return;

    if (auto* resources = SVGResourcesCache::cachedResourcesForRenderer(renderer)) {
        resources->removeClientFromCacheAndMarkForInvalidation(renderer);
        resourcesCacheFromRenderer(renderer).removeResourcesFromRenderer(renderer);
    }
}

void SVGResourcesCache::resourceDestroyed(LegacyRenderSVGResourceContainer& resource)
{
    // While LBSE does not make use of SVGResourcesCache, we might get here after switching from legacy to LBSE
    // and destructing the legacy tree -- when LBSE is already activated - don't assert here that this is not reached.
    if (resource.document().settings().layerBasedSVGEngineEnabled())
        return;

    auto& cache = resourcesCacheFromRenderer(resource);

    // The resource itself may have clients, that need to be notified.
    cache.removeResourcesFromRenderer(resource);

    for (auto& it : cache.m_cache) {
        if (it.value->resourceDestroyed(resource)) {
            // Mark users of destroyed resources as pending resolution based on the id of the old resource.
            Ref clientElement = *it.key->element();
            clientElement->treeScopeForSVGReferences().addPendingSVGResource(resource.element().getIdAttribute(), downcast<SVGElement>(clientElement.get()));
        }
    }
}

SVGResourcesCache::SetStyleForScope::SetStyleForScope(RenderElement& renderer, const RenderStyle& scopedStyle, const RenderStyle& newStyle)
    : m_renderer(renderer)
    , m_scopedStyle(scopedStyle)
    , m_needsNewStyle(scopedStyle != newStyle && rendererCanHaveResources(renderer))
{
    setStyle(newStyle);
}

SVGResourcesCache::SetStyleForScope::~SetStyleForScope()
{
    setStyle(m_scopedStyle);
}

void SVGResourcesCache::SetStyleForScope::setStyle(const RenderStyle& style)
{
    if (!m_needsNewStyle)
        return;

    // FIXME: Check if a similar mechanism is needed for LBSE + text rendering.
    if (m_renderer.document().settings().layerBasedSVGEngineEnabled())
        return;

    auto& cache = resourcesCacheFromRenderer(m_renderer);
    cache.removeResourcesFromRenderer(m_renderer);
    cache.addResourcesFromRenderer(m_renderer, style);
}

}
