/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "SVGResourcesCycleSolver.h"

#include "LegacyRenderSVGResourceFilter.h"
#include "LegacyRenderSVGResourceMarker.h"
#include "LegacyRenderSVGResourceMasker.h"
#include "Logging.h"
#include "RenderAncestorIterator.h"
#include "SVGGradientElement.h"
#include "SVGPatternElement.h"
#include "SVGResources.h"
#include "SVGResourcesCache.h"
#include <wtf/Scope.h>

namespace WebCore {

bool SVGResourcesCycleSolver::resourceContainsCycles(LegacyRenderSVGResourceContainer& resource,
    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer>& activeResources, SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer>& acyclicResources)
{
    if (acyclicResources.contains(resource))
        return false;

    activeResources.add(resource);
    auto activeResourceAcquisition = WTF::makeScopeExit([&]() {
        activeResources.remove(resource);
    });

    RenderObject* node = &resource;
    while (node) {
        if (node != &resource && node->isLegacyRenderSVGResourceContainer()) {
            node = node->nextInPreOrderAfterChildren(&resource);
            continue;
        }
        if (auto* element = dynamicDowncast<RenderElement>(*node)) {
            if (auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*element)) {
                SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> resourceSet;
                resources->buildSetOfResources(resourceSet);

                for (auto& resource : resourceSet) {
                    if (activeResources.contains(resource) || resourceContainsCycles(resource, activeResources, acyclicResources))
                        return true;
                }
            }
        }

        node = node->nextInPreOrder(&resource);
    }

    acyclicResources.add(resource);
    return false;
}

void SVGResourcesCycleSolver::resolveCycles(RenderElement& renderer, SVGResources& resources)
{
    // Verify that LBSE does not make use of SVGResourcesCache.
    if (renderer.document().settings().layerBasedSVGEngineEnabled())
        RELEASE_ASSERT_NOT_REACHED();

    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> localResources;
    resources.buildSetOfResources(localResources);

    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> activeResources;
    SingleThreadWeakHashSet<LegacyRenderSVGResourceContainer> acyclicResources;

    if (auto* container = dynamicDowncast<LegacyRenderSVGResourceContainer>(renderer))
        activeResources.add(*container);

    // The job of this function is to determine wheter any of the 'resources' associated with the given 'renderer'
    // references us (or whether any of its kids references us) -> that's a cycle, we need to find and break it.
    for (auto& resource : localResources) {
        if (activeResources.contains(resource) || resourceContainsCycles(resource, activeResources, acyclicResources))
            breakCycle(resource, resources);
    }
}

void SVGResourcesCycleSolver::breakCycle(LegacyRenderSVGResourceContainer& resourceLeadingToCycle, SVGResources& resources)
{
    if (&resourceLeadingToCycle == resources.linkedResource()) {
        resources.resetLinkedResource();
        return;
    }

    switch (resourceLeadingToCycle.resourceType()) {
    case MaskerResourceType:
        ASSERT(&resourceLeadingToCycle == resources.masker());
        resources.resetMasker();
        break;
    case MarkerResourceType:
        ASSERT(&resourceLeadingToCycle == resources.markerStart() || &resourceLeadingToCycle == resources.markerMid() || &resourceLeadingToCycle == resources.markerEnd());
        if (resources.markerStart() == &resourceLeadingToCycle)
            resources.resetMarkerStart();
        if (resources.markerMid() == &resourceLeadingToCycle)
            resources.resetMarkerMid();
        if (resources.markerEnd() == &resourceLeadingToCycle)
            resources.resetMarkerEnd();
        break;
    case PatternResourceType:
    case LinearGradientResourceType:
    case RadialGradientResourceType:
        ASSERT(&resourceLeadingToCycle == resources.fill() || &resourceLeadingToCycle == resources.stroke());
        if (resources.fill() == &resourceLeadingToCycle)
            resources.resetFill();
        if (resources.stroke() == &resourceLeadingToCycle)
            resources.resetStroke();
        break;
    case FilterResourceType:
        ASSERT(&resourceLeadingToCycle == resources.filter());
        resources.resetFilter();
        break;
    case ClipperResourceType:
        ASSERT(&resourceLeadingToCycle == resources.clipper());
        resources.resetClipper();
        break;
    case SolidColorResourceType:
        ASSERT_NOT_REACHED();
        break;
    }
}

}
