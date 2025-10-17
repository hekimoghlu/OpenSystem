/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "RenderSVGResourceContainer.h"

#include "RenderLayer.h"
#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGRoot.h"
#include "SVGElementTypeHelpers.h"
#include "SVGResourceElementClient.h"
#include "SVGVisitedElementTracking.h"
#include <wtf/SetForScope.h>
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceContainer);

RenderSVGResourceContainer::RenderSVGResourceContainer(Type type, SVGElement& element, RenderStyle&& style)
    : RenderSVGHiddenContainer(type, element, WTFMove(style), SVGModelObjectFlag::IsResourceContainer)
    , m_id(element.getIdAttribute())
{
    ASSERT(isRenderSVGResourceContainer());
}

RenderSVGResourceContainer::~RenderSVGResourceContainer() = default;

void RenderSVGResourceContainer::willBeDestroyed()
{
    m_registered = false;
    RenderSVGHiddenContainer::willBeDestroyed();
}

void RenderSVGResourceContainer::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderSVGHiddenContainer::styleDidChange(diff, oldStyle);

    if (!m_registered) {
        m_registered = true;
        registerResource();
    }
}

void RenderSVGResourceContainer::idChanged()
{
    // Remove old id, that is guaranteed to be present in cache.
    m_id = element().getIdAttribute();

    registerResource();
}

static inline void notifyResourceChanged(SVGElement& element)
{
    static NeverDestroyed<SVGVisitedElementTracking::VisitedSet> s_visitedSet;

    SVGVisitedElementTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(element))
        return;

    SVGVisitedElementTracking::Scope recursionScope(recursionTracking, element);

    for (auto& cssClient : element.referencingCSSClients()) {
        if (!cssClient)
            continue;
        cssClient->resourceChanged(element);
    }
}

void RenderSVGResourceContainer::registerResource()
{
    Ref treeScope = this->treeScopeForSVGReferences();
    if (!treeScope->isIdOfPendingSVGResource(m_id))
        return;

    auto elements = copyToVectorOf<Ref<SVGElement>>(treeScope->removePendingSVGResource(m_id));
    for (auto& element : elements) {
        ASSERT(element->hasPendingResources());
        treeScope->clearHasPendingSVGResourcesIfPossible(element);
        notifyResourceChanged(element.get());
    }
}

void RenderSVGResourceContainer::repaintAllClients() const
{
    Ref svgElement = element();
    notifyResourceChanged(svgElement.get());
}

}
