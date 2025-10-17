/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

#include "HeapInlines.h"
#include "WeakGCMap.h"
#include "WeakInlines.h"
#include <wtf/IterationStatus.h>

namespace JSC {

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
inline WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::WeakGCMap(VM& vm)
    : m_vm(vm)
{
    vm.heap.registerWeakGCHashTable(this);
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
inline WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::~WeakGCMap()
{
    m_vm.heap.unregisterWeakGCHashTable(this);
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
inline typename WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::iterator WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::find(const KeyType& key)
{
    iterator it = m_map.find(key);
    iterator end = m_map.end();
    if (it != end && !it->value) // Found a zombie value.
        return end;
    return it;
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
inline typename WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::const_iterator WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::find(const KeyType& key) const
{
    return const_cast<WeakGCMap*>(this)->find(key);
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
inline bool WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::contains(const KeyType& key) const
{
    return find(key) != m_map.end();
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
NEVER_INLINE void WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::pruneStaleEntries()
{
    m_map.removeIf([](const typename HashMapType::KeyValuePairType& entry) {
        return !entry.value;
    });
}

template<typename KeyArg, typename ValueArg, typename HashArg, typename KeyTraitsArg>
template<typename Func>
inline void WeakGCMap<KeyArg, ValueArg, HashArg, KeyTraitsArg>::forEach(Func func)
{
    ASSERT(m_vm.heap.isDeferred());
    for (auto& entry : m_map) {
        if (entry.value) {
            if (func(entry.value.get()) == IterationStatus::Done)
                return;
        }
    }
}

} // namespace JSC
