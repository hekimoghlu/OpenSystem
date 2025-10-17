/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include "LayerPool.h"

#include "Logging.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LayerPool);

static constexpr Seconds capacityDecayTime { 5_s };

LayerPool::LayerPool()
    : m_maxBytesForPool(48 * 1024 * 1024)
    , m_pruneTimer(*this, &LayerPool::pruneTimerFired)
{
    RELEASE_ASSERT(isMainThread());
    allLayerPools().add(this);
}

LayerPool::~LayerPool()
{
    RELEASE_ASSERT(isMainThread());
    allLayerPools().remove(this);
}

UncheckedKeyHashSet<CheckedPtr<LayerPool>>& LayerPool::allLayerPools()
{
    RELEASE_ASSERT(isMainThread());
    static NeverDestroyed<UncheckedKeyHashSet<CheckedPtr<LayerPool>>> allLayerPools;
    return allLayerPools.get();
}

unsigned LayerPool::backingStoreBytesForSize(const IntSize& size)
{
    return size.area() * 4;
}

LayerPool::LayerList& LayerPool::listOfLayersWithSize(const IntSize& size, AccessType accessType)
{
    UncheckedKeyHashMap<IntSize, LayerList>::iterator it = m_reuseLists.find(size);
    if (it == m_reuseLists.end()) {
        it = m_reuseLists.add(size, LayerList()).iterator;
        m_sizesInPruneOrder.append(size);
    } else if (accessType == MarkAsUsed) {
        m_sizesInPruneOrder.remove(m_sizesInPruneOrder.reverseFind(size));
        m_sizesInPruneOrder.append(size);
    }
    return it->value;
}

void LayerPool::addLayer(const RefPtr<PlatformCALayer>& layer)
{
    RELEASE_ASSERT(isMainThread());
    IntSize layerSize(expandedIntSize(layer->bounds().size()));
    if (!canReuseLayerWithSize(layerSize))
        return;

    listOfLayersWithSize(layerSize).prepend(layer);
    m_totalBytes += backingStoreBytesForSize(layerSize);
    
    m_lastAddTime = MonotonicTime::now();
    schedulePrune();
}

RefPtr<PlatformCALayer> LayerPool::takeLayerWithSize(const IntSize& size)
{
    RELEASE_ASSERT(isMainThread());
    if (!canReuseLayerWithSize(size))
        return nullptr;
    LayerList& reuseList = listOfLayersWithSize(size, MarkAsUsed);
    if (reuseList.isEmpty())
        return nullptr;
    m_totalBytes -= backingStoreBytesForSize(size);
    return reuseList.takeFirst();
}
    
unsigned LayerPool::decayedCapacity() const
{
    // Decay to one quarter over capacityDecayTime
    Seconds timeSinceLastAdd = MonotonicTime::now() - m_lastAddTime;
    if (timeSinceLastAdd > capacityDecayTime)
        return m_maxBytesForPool / 4;
    float decayProgess = float(timeSinceLastAdd / capacityDecayTime);
    return m_maxBytesForPool / 4 + m_maxBytesForPool * 3 / 4 * (1 - decayProgess);
}

void LayerPool::schedulePrune()
{
    if (m_pruneTimer.isActive())
        return;
    m_pruneTimer.startOneShot(1_s);
}

void LayerPool::pruneTimerFired()
{
    RELEASE_ASSERT(isMainThread());
    unsigned shrinkTo = decayedCapacity();
    while (m_totalBytes > shrinkTo) {
        RELEASE_ASSERT(!m_sizesInPruneOrder.isEmpty());
        IntSize sizeToDrop = m_sizesInPruneOrder.first();
        auto it = m_reuseLists.find(sizeToDrop);
        RELEASE_ASSERT(it != m_reuseLists.end());
        LayerList& oldestReuseList = it->value;
        if (oldestReuseList.isEmpty()) {
            m_reuseLists.remove(sizeToDrop);
            m_sizesInPruneOrder.remove(0);
            continue;
        }

        ASSERT(m_totalBytes >= backingStoreBytesForSize(sizeToDrop));
        m_totalBytes -= backingStoreBytesForSize(sizeToDrop);
        // The last element in the list is the oldest, hence most likely not to
        // still have a backing store.
        oldestReuseList.removeLast();
    }
    if (MonotonicTime::now() - m_lastAddTime <= capacityDecayTime)
        schedulePrune();
}

void LayerPool::drain()
{
    RELEASE_ASSERT(isMainThread());
    m_reuseLists.clear();
    m_sizesInPruneOrder.clear();
    m_totalBytes = 0;
}

}
