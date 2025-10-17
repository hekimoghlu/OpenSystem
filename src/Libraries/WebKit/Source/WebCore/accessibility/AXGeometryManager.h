/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
#include "AXCoreObject.h"
#include "IntRectHash.h"
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class AXObjectCache;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AXGeometryManager);
class AXGeometryManager final : public ThreadSafeRefCounted<AXGeometryManager> {
    WTF_MAKE_NONCOPYABLE(AXGeometryManager);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(AXGeometryManager);
public:
    explicit AXGeometryManager(AXObjectCache&);
    AXGeometryManager();
    static Ref<AXGeometryManager> create(AXObjectCache& cache)
    {
        return adoptRef(*new AXGeometryManager(cache));
    }
    ~AXGeometryManager();

    void willUpdateObjectRegions();
    void scheduleObjectRegionsUpdate(bool /* scheduleImmediately */);

    void cacheRect(std::optional<AXID>, IntRect&&);
    // std::nullopt if there is no cached rect for the given ID (i.e. because it hasn't been cached yet via paint or otherwise, or cannot be painted / cached at all).
    std::optional<IntRect> cachedRectForID(AXID);

    void remove(AXID axID) { m_cachedRects.remove(axID); }

#if PLATFORM(MAC)
    void initializePrimaryScreenRect();
    FloatRect primaryScreenRect();
#endif

private:
    void updateObjectRegionsTimerFired() { scheduleRenderingUpdate(); }
    void scheduleRenderingUpdate();

    // The cache that owns this instance.
    WeakPtr<AXObjectCache> m_cache;
    UncheckedKeyHashMap<AXID, IntRect> m_cachedRects;
    Timer m_updateObjectRegionsTimer;

#if PLATFORM(MAC)
    FloatRect m_primaryScreenRect WTF_GUARDED_BY_LOCK(m_lock);
    Lock m_lock;
#endif
};

} // namespace WebCore

#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
