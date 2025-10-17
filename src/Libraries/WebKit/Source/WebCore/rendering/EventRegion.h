/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

#include "AffineTransform.h"
#include "FloatRoundedRect.h"
#include "IntRect.h"
#include "IntRectHash.h"
#include "InteractionRegion.h"
#include "Node.h"
#include "Region.h"
#include "RegionContext.h"
#include "RenderStyleConstants.h"
#include "TouchAction.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class EventRegion;
class Path;
class RenderObject;
class RenderStyle;

class EventRegionContext final : public RegionContext {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(EventRegionContext, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(EventRegionContext);
public:
    WEBCORE_EXPORT explicit EventRegionContext(EventRegion&);
    WEBCORE_EXPORT virtual ~EventRegionContext();

    bool isEventRegionContext() const final { return true; }

    WEBCORE_EXPORT void unite(const FloatRoundedRect&, RenderObject&, const RenderStyle&, bool overrideUserModifyIsEditable = false);
    bool contains(const IntRect&) const;

#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    void uniteInteractionRegions(RenderObject&, const FloatRect&, const FloatSize&, const std::optional<AffineTransform>&);
    bool shouldConsolidateInteractionRegion(RenderObject&, const IntRect&, const ElementIdentifier&);
    void convertGuardContainersToInterationIfNeeded(float minimumCornerRadius);
    void removeSuperfluousInteractionRegions();
    void shrinkWrapInteractionRegions();
    void copyInteractionRegionsToEventRegion(float minimumCornerRadius);
#endif

private:
    EventRegion& m_eventRegion;

#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    Vector<InteractionRegion> m_interactionRegions;
    UncheckedKeyHashMap<IntRect, InteractionRegion::ContentHint> m_interactionRectsAndContentHints;
    UncheckedKeyHashSet<IntRect> m_occlusionRects;
    enum class Inflated : bool { No, Yes };
    UncheckedKeyHashMap<IntRect, Inflated> m_guardRects;
    UncheckedKeyHashSet<ElementIdentifier> m_containerRemovalCandidates;
    UncheckedKeyHashSet<ElementIdentifier> m_containersToRemove;
    UncheckedKeyHashMap<ElementIdentifier, Vector<InteractionRegion>> m_discoveredRegionsByElement;
#endif
};

class EventRegion {
public:
    WEBCORE_EXPORT EventRegion();
    WEBCORE_EXPORT EventRegion(Region&&
#if ENABLE(TOUCH_ACTION_REGIONS)
    , Vector<WebCore::Region> touchActionRegions
#endif
#if ENABLE(WHEEL_EVENT_REGIONS)
    , WebCore::Region wheelEventListenerRegion
    , WebCore::Region nonPassiveWheelEventListenerRegion
#endif
#if ENABLE(EDITABLE_REGION)
    , std::optional<WebCore::Region>
#endif
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    , Vector<WebCore::InteractionRegion>
#endif
    );

    EventRegionContext makeContext() { return EventRegionContext(*this); }

    bool isEmpty() const { return m_region.isEmpty(); }

    friend bool operator==(const EventRegion&, const EventRegion&) = default;

    void unite(const Region&, RenderObject&, const RenderStyle&, bool overrideUserModifyIsEditable = false);
    void translate(const IntSize&);

    bool contains(const IntPoint& point) const { return m_region.contains(point); }
    bool contains(const IntRect& rect) const { return m_region.contains(rect); }
    bool intersects(const IntRect& rect) const { return m_region.intersects(rect); }

    const Region& region() const { return m_region; }

#if ENABLE(TOUCH_ACTION_REGIONS)
    bool hasTouchActions() const { return !m_touchActionRegions.isEmpty(); }
    WEBCORE_EXPORT OptionSet<TouchAction> touchActionsForPoint(const IntPoint&) const;
    const Region* regionForTouchAction(TouchAction) const;
#endif

#if ENABLE(WHEEL_EVENT_REGIONS)
    WEBCORE_EXPORT OptionSet<EventListenerRegionType> eventListenerRegionTypesForPoint(const IntPoint&) const;
    const Region& eventListenerRegionForType(EventListenerRegionType) const;
#endif

#if ENABLE(EDITABLE_REGION)
    void ensureEditableRegion();
    bool hasEditableRegion() const { return m_editableRegion.has_value(); }
    WEBCORE_EXPORT bool containsEditableElementsInRect(const IntRect&) const;
    Vector<IntRect, 1> rectsForEditableElements() const { return m_editableRegion ? m_editableRegion->rects() : Vector<IntRect, 1> { }; }
#endif

    void dump(TextStream&) const;

#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    const Vector<InteractionRegion>& interactionRegions() const { return m_interactionRegions; }
    void appendInteractionRegions(const Vector<InteractionRegion>&);
    void clearInteractionRegions();
#endif

private:
    friend struct IPC::ArgumentCoder<EventRegion, void>;
#if ENABLE(TOUCH_ACTION_REGIONS)
    void uniteTouchActions(const Region&, OptionSet<TouchAction>);
#endif
    void uniteEventListeners(const Region&, OptionSet<EventListenerRegionType>);

    Region m_region;
#if ENABLE(TOUCH_ACTION_REGIONS)
    Vector<Region> m_touchActionRegions;
#endif
#if ENABLE(WHEEL_EVENT_REGIONS)
    Region m_wheelEventListenerRegion;
    Region m_nonPassiveWheelEventListenerRegion;
#endif
#if ENABLE(EDITABLE_REGION)
    std::optional<Region> m_editableRegion;
#endif
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    Vector<InteractionRegion> m_interactionRegions;
#endif
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const EventRegion&);

#if ENABLE(EDITABLE_REGION)

inline void EventRegion::ensureEditableRegion()
{
    if (!m_editableRegion)
        m_editableRegion.emplace();
}

#endif

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::EventRegionContext)
    static bool isType(const WebCore::RegionContext& regionContext) { return regionContext.isEventRegionContext(); }
SPECIALIZE_TYPE_TRAITS_END()
