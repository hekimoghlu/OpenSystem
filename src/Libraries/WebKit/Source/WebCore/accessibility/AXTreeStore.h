/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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

#include "AXCoreObject.h"
#include "ActivityState.h"
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
class AXIsolatedTree;
#endif
class AXObjectCache;

using AXTreePtr = std::variant<std::nullptr_t, WeakPtr<AXObjectCache>
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    , RefPtr<AXIsolatedTree>
#endif
>;

using AXTreeWeakPtr = std::variant<WeakPtr<AXObjectCache>
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    , ThreadSafeWeakPtr<AXIsolatedTree>
#endif
>;

AXTreePtr axTreeForID(AXID);
WEBCORE_EXPORT AXTreePtr findAXTree(Function<bool(AXTreePtr)>&&);

template<typename T>
class AXTreeStore {
    // For now, we just disable direct instantiations of this class because it is not
    // needed. Subclasses are expected to declare their own WTF_MAKE_TZONE_ALLOCATED.
    WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(AXTreeStore);
    WTF_MAKE_NONCOPYABLE(AXTreeStore);
    friend WEBCORE_EXPORT AXTreePtr findAXTree(Function<bool(AXTreePtr)>&&);
public:
    AXID treeID() const { return m_id; }
    static WeakPtr<AXObjectCache> axObjectCacheForID(std::optional<AXID>);
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    static RefPtr<AXIsolatedTree> isolatedTreeForID(std::optional<AXID>);
#endif

protected:
    AXTreeStore(AXID axID = generateNewID())
        : m_id(axID)
    { }

    static void set(AXID, const AXTreeWeakPtr&);
    static void add(AXID, const AXTreeWeakPtr&);
    static void remove(AXID);
    static bool contains(AXID);

    static AXID generateNewID();
    const AXID m_id;
    static Lock s_storeLock;
private:
    static UncheckedKeyHashMap<AXID, WeakPtr<AXObjectCache>>& liveTreeMap();
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    static UncheckedKeyHashMap<AXID, ThreadSafeWeakPtr<AXIsolatedTree>>& isolatedTreeMap() WTF_REQUIRES_LOCK(s_storeLock);
#endif
};

template<typename T>
inline void AXTreeStore<T>::set(AXID axID, const AXTreeWeakPtr& tree)
{
    ASSERT(isMainThread());

    switchOn(tree,
        [&] (const WeakPtr<AXObjectCache>& typedTree) {
            liveTreeMap().set(axID, typedTree);
        }
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
        , [&] (const ThreadSafeWeakPtr<AXIsolatedTree>& typedTree) {
            Locker locker { s_storeLock };
            isolatedTreeMap().set(axID, typedTree.get());
        }
#endif
    );
}

template<typename T>
inline void AXTreeStore<T>::add(AXID axID, const AXTreeWeakPtr& tree)
{
    ASSERT(isMainThread());

    switchOn(tree,
        [&] (const WeakPtr<AXObjectCache>& typedTree) {
            liveTreeMap().add(axID, typedTree);
        }
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
        , [&] (const ThreadSafeWeakPtr<AXIsolatedTree>& typedTree) {
            Locker locker { s_storeLock };
            isolatedTreeMap().add(axID, typedTree.get());
        }
#endif
    );
}

template<typename T>
inline void AXTreeStore<T>::remove(AXID axID)
{
    if (isMainThread()) {
        liveTreeMap().remove(axID);
        return;
    }
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Locker locker { s_storeLock };
    isolatedTreeMap().remove(axID);
#endif
}

template<typename T>
inline bool AXTreeStore<T>::contains(AXID axID)
{
    if (isMainThread())
        return liveTreeMap().contains(axID);
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Locker locker { s_storeLock };
    return isolatedTreeMap().contains(axID);
#endif
}

template<typename T>
inline WeakPtr<AXObjectCache> AXTreeStore<T>::axObjectCacheForID(std::optional<AXID> axID)
{
    return axID ? liveTreeMap().get(*axID) : nullptr;
}

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
template<typename T>
inline RefPtr<AXIsolatedTree> AXTreeStore<T>::isolatedTreeForID(std::optional<AXID> axID)
{
    if (!axID)
        return nullptr;

    Locker locker { s_storeLock };
    return isolatedTreeMap().get(*axID).get();
}
#endif

template<typename T>
inline UncheckedKeyHashMap<AXID, WeakPtr<AXObjectCache>>& AXTreeStore<T>::liveTreeMap()
{
    ASSERT(isMainThread());

    static NeverDestroyed<UncheckedKeyHashMap<AXID, WeakPtr<AXObjectCache>>> map;
    return map;
}

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
template<typename T>
inline UncheckedKeyHashMap<AXID, ThreadSafeWeakPtr<AXIsolatedTree>>& AXTreeStore<T>::isolatedTreeMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<AXID, ThreadSafeWeakPtr<AXIsolatedTree>>> map;
    return map;
}
#endif

template<typename T>
inline AXID AXTreeStore<T>::generateNewID()
{
    ASSERT(isMainThread());

    std::optional<AXID> axID;
    do {
        axID = AXID::generate();
    } while (liveTreeMap().contains(*axID));
    return *axID;
}

template<typename T>
Lock AXTreeStore<T>::s_storeLock;

inline AXTreePtr axTreeForID(std::optional<AXID> axID)
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainThread())
        return AXTreeStore<AXIsolatedTree>::isolatedTreeForID(axID);
#endif
    return AXTreeStore<AXObjectCache>::axObjectCacheForID(axID);
}

} // namespace WebCore
