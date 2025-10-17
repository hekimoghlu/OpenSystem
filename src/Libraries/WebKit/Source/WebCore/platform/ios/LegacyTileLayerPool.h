/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

#if PLATFORM(IOS_FAMILY)

#include "IntSize.h"
#include "IntSizeHash.h"
#include "Timer.h"
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/RetainPtr.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>

@class LegacyTileLayer;

namespace WebCore {

class LegacyTileLayerPool {
    WTF_MAKE_NONCOPYABLE(LegacyTileLayerPool);
public:
    static LegacyTileLayerPool* sharedPool();

    void addLayer(const RetainPtr<LegacyTileLayer>&);
    RetainPtr<LegacyTileLayer> takeLayerWithSize(const IntSize&);

    // The maximum size of all queued layers in bytes.
    unsigned capacity() const { return m_capacity; }
    void setCapacity(unsigned);
    void drain();

    unsigned decayedCapacity() const;

    static unsigned bytesBackingLayerWithPixelSize(const IntSize&);

private:
    LegacyTileLayerPool();

    typedef Deque<RetainPtr<LegacyTileLayer> > LayerList;

    bool canReuseLayerWithSize(const IntSize& size) const { return m_capacity && !size.isEmpty(); }
    void schedulePrune();
    void prune();
    typedef enum { LeaveUnchanged, MarkAsUsed } AccessType;
    LayerList& listOfLayersWithSize(const IntSize&, AccessType = LeaveUnchanged);

    UncheckedKeyHashMap<IntSize, LayerList> m_reuseLists;
    // Ordered by recent use. The last size is the most recently used.
    Vector<IntSize> m_sizesInPruneOrder;
    unsigned m_totalBytes;
    unsigned m_capacity;
    Lock m_layerPoolMutex;

    WallTime m_lastAddTime;
    bool m_needsPrune;

    friend NeverDestroyed<LegacyTileLayerPool>;
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
