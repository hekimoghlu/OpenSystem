/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#ifndef LayerPool_h
#define LayerPool_h

#include "IntSize.h"
#include "IntSizeHash.h"
#include "PlatformCALayer.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
    
class LayerPool final : public CanMakeCheckedPtr<LayerPool> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LayerPool, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LayerPool);
    WTF_MAKE_NONCOPYABLE(LayerPool);
public:
    WEBCORE_EXPORT LayerPool();
    WEBCORE_EXPORT ~LayerPool();

    static UncheckedKeyHashSet<CheckedPtr<LayerPool>>& allLayerPools();
    
    void addLayer(const RefPtr<PlatformCALayer>&);
    RefPtr<PlatformCALayer> takeLayerWithSize(const IntSize&);

    void drain();

    // The maximum size of all queued layers in bytes.
    unsigned capacity() const { return m_maxBytesForPool; }

private:
    typedef Deque<RefPtr<PlatformCALayer>> LayerList;

    unsigned decayedCapacity() const;

    bool canReuseLayerWithSize(const IntSize& size) const { return m_maxBytesForPool && !size.isEmpty(); }
    void schedulePrune();
    void pruneTimerFired();

    typedef enum { LeaveUnchanged, MarkAsUsed } AccessType;
    LayerList& listOfLayersWithSize(const IntSize&, AccessType = LeaveUnchanged);

    static unsigned backingStoreBytesForSize(const IntSize&);

    UncheckedKeyHashMap<IntSize, LayerList> m_reuseLists;
    // Ordered by recent use. The last size is the most recently used.
    Vector<IntSize> m_sizesInPruneOrder;
    unsigned m_totalBytes { 0 };
    unsigned m_maxBytesForPool;

    Timer m_pruneTimer;

    MonotonicTime m_lastAddTime;
};

}

#endif
