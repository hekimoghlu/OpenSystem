/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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

#include "LegacyRenderSVGHiddenContainer.h"
#include "LegacyRenderSVGResource.h"
#include "SVGDocumentExtensions.h"
#include <wtf/WeakHashSet.h>

namespace WebCore {

class RenderLayer;

class LegacyRenderSVGResourceContainer : public LegacyRenderSVGHiddenContainer, public LegacyRenderSVGResource {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceContainer);
public:
    virtual ~LegacyRenderSVGResourceContainer();

    void layout() override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;

    static float computeTextPaintingScale(const RenderElement&);
    static AffineTransform transformOnNonScalingStroke(RenderObject*, const AffineTransform& resourceTransform);

    void removeClientFromCacheAndMarkForInvalidation(RenderElement&, bool markForInvalidation = true) override;
    void removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool markForInvalidation, SingleThreadWeakHashSet<RenderObject>* visitedRenderers) override;

    void idChanged();
    void markAllClientsForRepaint();
    void addClientRenderLayer(RenderLayer&);
    void removeClientRenderLayer(RenderLayer&);
    void markAllClientLayersForInvalidation();

protected:
    LegacyRenderSVGResourceContainer(Type, SVGElement&, RenderStyle&&);

    enum InvalidationMode {
        LayoutAndBoundariesInvalidation,
        BoundariesInvalidation,
        RepaintInvalidation,
        ParentOnlyInvalidation
    };

    // Used from the invalidateClient/invalidateClients methods from classes, inheriting from us.
    virtual bool selfNeedsClientInvalidation() const { return everHadLayout() && selfNeedsLayout(); }

    void markAllClientsForInvalidation(InvalidationMode);
    void markAllClientsForInvalidationIfNeeded(InvalidationMode, SingleThreadWeakHashSet<RenderObject>* visitedRenderers);
    void markClientForInvalidation(RenderObject&, InvalidationMode);

private:
    friend class SVGResourcesCache;
    void addClient(RenderElement&);
    void removeClient(RenderElement&);

    void willBeDestroyed() final;
    void registerResource();

    AtomString m_id;
    SingleThreadWeakHashSet<RenderElement> m_clients;
    SingleThreadWeakHashSet<RenderLayer> m_clientLayers;
    bool m_registered { false };
    bool m_isInvalidating { false };
};

inline LegacyRenderSVGResourceContainer* getRenderSVGResourceContainerById(TreeScope& treeScope, const AtomString& id)
{
    if (id.isEmpty())
        return nullptr;

    if (LegacyRenderSVGResourceContainer* renderResource = treeScope.lookupLegacySVGResoureById(id))
        return renderResource;

    return nullptr;
}

template<typename Renderer>
Renderer* getRenderSVGResourceById(TreeScope& treeScope, const AtomString& id)
{
    // Using the LegacyRenderSVGResource type here avoids ambiguous casts for types that
    // descend from both RenderObject and LegacyRenderSVGResourceContainer.
    LegacyRenderSVGResource* container = getRenderSVGResourceContainerById(treeScope, id);
    return dynamicDowncast<Renderer>(container);
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGResourceContainer, isLegacyRenderSVGResourceContainer())
