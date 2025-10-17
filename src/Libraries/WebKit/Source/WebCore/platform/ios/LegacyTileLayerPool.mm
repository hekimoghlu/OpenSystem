/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#import "config.h"
#import "LegacyTileLayerPool.h"

#if PLATFORM(IOS_FAMILY)

#import "LegacyTileGrid.h"
#import "LegacyTileLayer.h"
#import "Logging.h"
#import <wtf/MemoryPressureHandler.h>
#import <wtf/NeverDestroyed.h>

namespace WebCore {

static const Seconds capacityDecayTime { 5_s };

LegacyTileLayerPool::LegacyTileLayerPool()
    : m_totalBytes(0)
    , m_capacity(0)
    , m_needsPrune(false)
{
}

LegacyTileLayerPool* LegacyTileLayerPool::sharedPool()
{
    static NeverDestroyed<LegacyTileLayerPool> sharedPool;
    return &sharedPool.get();
}

unsigned LegacyTileLayerPool::bytesBackingLayerWithPixelSize(const IntSize& size)
{
    return size.area() * 4;
}

LegacyTileLayerPool::LayerList& LegacyTileLayerPool::listOfLayersWithSize(const IntSize& size, AccessType accessType)
{
    ASSERT(!m_layerPoolMutex.tryLock());
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

void LegacyTileLayerPool::addLayer(const RetainPtr<LegacyTileLayer>& layer)
{
    IntSize layerSize([layer frame].size);
    layerSize.scale([layer contentsScale]);
    if (!canReuseLayerWithSize(layerSize))
        return;

    if (MemoryPressureHandler::singleton().isUnderMemoryPressure()) {
        LOG(MemoryPressure, "Under memory pressure: %s, totalBytes: %d", __PRETTY_FUNCTION__, m_totalBytes);
        return;
    }

    Locker locker { m_layerPoolMutex };
    listOfLayersWithSize(layerSize).prepend(layer);
    m_totalBytes += bytesBackingLayerWithPixelSize(layerSize);

    m_lastAddTime = WallTime::now();
    schedulePrune();
}

RetainPtr<LegacyTileLayer> LegacyTileLayerPool::takeLayerWithSize(const IntSize& size)
{
    if (!canReuseLayerWithSize(size))
        return nil;
    Locker locker { m_layerPoolMutex };
    LayerList& reuseList = listOfLayersWithSize(size, MarkAsUsed);
    if (reuseList.isEmpty())
        return nil;
    m_totalBytes -= bytesBackingLayerWithPixelSize(size);
    return reuseList.takeFirst();
}

void LegacyTileLayerPool::setCapacity(unsigned capacity)
{
    Locker reuseLocker { m_layerPoolMutex };
    if (capacity < m_capacity)
        schedulePrune();
    m_capacity = capacity;
}
    
unsigned LegacyTileLayerPool::decayedCapacity() const
{
    // Decay to one quarter over capacityDecayTime
    Seconds timeSinceLastAdd = WallTime::now() - m_lastAddTime;
    if (timeSinceLastAdd > capacityDecayTime)
        return m_capacity / 4;
    float decayProgess = float(timeSinceLastAdd / capacityDecayTime);
    return m_capacity / 4 + m_capacity * 3 / 4 * (1.f - decayProgess);
}

void LegacyTileLayerPool::schedulePrune()
{
    ASSERT(!m_layerPoolMutex.tryLock());
    if (m_needsPrune)
        return;
    m_needsPrune = true;
    dispatch_time_t nextPruneTime = dispatch_time(DISPATCH_TIME_NOW, 1 * NSEC_PER_SEC);
    dispatch_after(nextPruneTime, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        prune();
    });
}

void LegacyTileLayerPool::prune()
{
    Locker locker { m_layerPoolMutex };
    ASSERT(m_needsPrune);
    m_needsPrune = false;
    unsigned shrinkTo = decayedCapacity();
    while (m_totalBytes > shrinkTo) {
        ASSERT(!m_sizesInPruneOrder.isEmpty());
        IntSize sizeToDrop = m_sizesInPruneOrder.first();
        LayerList& oldestReuseList = m_reuseLists.find(sizeToDrop)->value;
        if (oldestReuseList.isEmpty()) {
            m_reuseLists.remove(sizeToDrop);
            m_sizesInPruneOrder.remove(0);
            continue;
        }
#if LOG_TILING
        NSLog(@"dropping layer of size %d x %d", sizeToDrop.width(), sizeToDrop.height());
#endif
        m_totalBytes -= bytesBackingLayerWithPixelSize(sizeToDrop);
        // The last element in the list is the oldest, hence most likely not to
        // still have a backing store.
        oldestReuseList.removeLast();
    }
    if (WallTime::now() - m_lastAddTime <= capacityDecayTime)
        schedulePrune();
}

void LegacyTileLayerPool::drain()
{
    Locker reuseLocker { m_layerPoolMutex };
    m_reuseLists.clear();
    m_sizesInPruneOrder.clear();
    m_totalBytes = 0;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
