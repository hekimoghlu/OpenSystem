/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

#include "DebugOverlayRegions.h"
#include "LocalFrame.h"
#include <wtf/HashMap.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashMap.h>

namespace WebCore {

class Page;
class RegionOverlay;

class DebugPageOverlays {
public:
    static DebugPageOverlays& singleton();

    enum class RegionType : uint8_t {
        WheelEventHandlers,
        NonFastScrollableRegion,
        InteractionRegion,
        SiteIsolationRegion
    };
    static constexpr unsigned NumberOfRegionTypes = static_cast<unsigned>(RegionType::SiteIsolationRegion) + 1;

    static void didLayout(LocalFrame&);
    static void didChangeEventHandlers(LocalFrame&);
    static void doAfterUpdateRendering(Page&);

    static bool shouldPaintOverlayIntoLayerForRegionType(Page&, RegionType);

    WEBCORE_EXPORT static void settingsChanged(Page&);

private:
    static bool hasOverlays(Page&);

    void showRegionOverlay(Page&, RegionType);
    void hideRegionOverlay(Page&, RegionType);

    void updateRegionIfNecessary(Page&, RegionType);

    void regionChanged(LocalFrame&, RegionType);

    bool hasOverlaysForPage(Page&) const;
    
    void updateOverlayRegionVisibility(Page&, OptionSet<DebugOverlayRegions>);

    bool shouldPaintOverlayIntoLayer(Page&, RegionType) const;

    RegionOverlay* regionOverlayForPage(Page&, RegionType) const;
    RegionOverlay& ensureRegionOverlayForPage(Page&, RegionType);

    WeakHashMap<Page, Vector<RefPtr<RegionOverlay>>> m_pageRegionOverlays;

    static DebugPageOverlays* sharedDebugOverlays;
};

#define FAST_RETURN_IF_NO_OVERLAYS(page) if (LIKELY(!page || !hasOverlays(*page))) return;

inline bool DebugPageOverlays::hasOverlays(Page& page)
{
    if (!sharedDebugOverlays)
        return false;

    return sharedDebugOverlays->hasOverlaysForPage(page);
}

inline void DebugPageOverlays::didLayout(LocalFrame& frame)
{
    FAST_RETURN_IF_NO_OVERLAYS(frame.page());

    sharedDebugOverlays->regionChanged(frame, RegionType::WheelEventHandlers);
    sharedDebugOverlays->regionChanged(frame, RegionType::NonFastScrollableRegion);
    sharedDebugOverlays->regionChanged(frame, RegionType::InteractionRegion);
    sharedDebugOverlays->regionChanged(frame, RegionType::SiteIsolationRegion);
}

inline void DebugPageOverlays::didChangeEventHandlers(LocalFrame& frame)
{
    FAST_RETURN_IF_NO_OVERLAYS(frame.page());

    sharedDebugOverlays->regionChanged(frame, RegionType::WheelEventHandlers);
    sharedDebugOverlays->regionChanged(frame, RegionType::NonFastScrollableRegion);
    sharedDebugOverlays->regionChanged(frame, RegionType::InteractionRegion);
    sharedDebugOverlays->regionChanged(frame, RegionType::SiteIsolationRegion);
}

inline void DebugPageOverlays::doAfterUpdateRendering(Page& page)
{
    if (LIKELY(!hasOverlays(page)))
        return;

    sharedDebugOverlays->updateRegionIfNecessary(page, RegionType::WheelEventHandlers);
    sharedDebugOverlays->updateRegionIfNecessary(page, RegionType::NonFastScrollableRegion);
    sharedDebugOverlays->updateRegionIfNecessary(page, RegionType::InteractionRegion);
    sharedDebugOverlays->updateRegionIfNecessary(page, RegionType::SiteIsolationRegion);
}

inline bool DebugPageOverlays::shouldPaintOverlayIntoLayerForRegionType(Page& page, RegionType regionType)
{
    if (LIKELY(!hasOverlays(page)))
        return false;
    return sharedDebugOverlays->shouldPaintOverlayIntoLayer(page, regionType);
}

} // namespace WebCore
