/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#include "AXGeometryManager.h"

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
#include "AXIsolatedTree.h"
#include "AXObjectCache.h"
#include "Page.h"

#if PLATFORM(MAC)
#include "PlatformScreen.h"
#endif

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AXGeometryManager);

AXGeometryManager::AXGeometryManager(AXObjectCache& owningCache)
    : m_cache(owningCache)
    , m_updateObjectRegionsTimer(*this, &AXGeometryManager::updateObjectRegionsTimerFired)
{
}

AXGeometryManager::AXGeometryManager()
    : m_cache(nullptr)
    , m_updateObjectRegionsTimer(*this, &AXGeometryManager::updateObjectRegionsTimerFired)
{
}

AXGeometryManager::~AXGeometryManager()
{
    if (m_updateObjectRegionsTimer.isActive())
        m_updateObjectRegionsTimer.stop();
}

std::optional<IntRect> AXGeometryManager::cachedRectForID(AXID axID)
{
    auto rectIterator = m_cachedRects.find(axID);
    if (rectIterator != m_cachedRects.end())
        return rectIterator->value;
    return std::nullopt;
}

void AXGeometryManager::cacheRect(std::optional<AXID> axID, IntRect&& rect)
{
    // We shouldn't call this method on a geometry manager that has no page ID.
    ASSERT(m_cache->pageID());
    ASSERT(AXObjectCache::isIsolatedTreeEnabled());

    if (!axID)
        return;
    auto rectIterator = m_cachedRects.find(*axID);

    bool rectChanged = false;
    if (rectIterator != m_cachedRects.end()) {
        rectChanged = rectIterator->value != rect;
        if (rectChanged)
            rectIterator->value = rect;
    } else {
        rectChanged = true;
        m_cachedRects.set(*axID, rect);
    }

    if (!rectChanged)
        return;

    RefPtr tree = AXIsolatedTree::treeForPageID(*m_cache->pageID());
    if (!tree)
        return;
    tree->updateFrame(*axID, WTFMove(rect));
}

void AXGeometryManager::scheduleObjectRegionsUpdate(bool scheduleImmediately)
{
    if (LIKELY(!scheduleImmediately)) {
        if (!m_updateObjectRegionsTimer.isActive())
            m_updateObjectRegionsTimer.startOneShot(1_s);
        return;
    }

    if (m_updateObjectRegionsTimer.isActive())
        m_updateObjectRegionsTimer.stop();
    scheduleRenderingUpdate();
}

// The page is about to update accessibility object regions, so the deferred
// update queued with this timer is unnecessary.
void AXGeometryManager::willUpdateObjectRegions()
{
    if (m_updateObjectRegionsTimer.isActive())
        m_updateObjectRegionsTimer.stop();

    if (!m_cache)
        return;

    if (RefPtr tree = AXIsolatedTree::treeForPageID(m_cache->pageID()))
        tree->updateRootScreenRelativePosition();
}

void AXGeometryManager::scheduleRenderingUpdate()
{
    if (!m_cache || !m_cache->document())
        return;

    if (RefPtr page = m_cache->document()->page())
        page->scheduleRenderingUpdate(RenderingUpdateStep::AccessibilityRegionUpdate);
}

#if PLATFORM(MAC)
void AXGeometryManager::initializePrimaryScreenRect()
{
    Locker locker { m_lock };
    m_primaryScreenRect = screenRectForPrimaryScreen();
}

FloatRect AXGeometryManager::primaryScreenRect()
{
    Locker locker { m_lock };
    return m_primaryScreenRect;
}
#endif

} // namespace WebCore

#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
