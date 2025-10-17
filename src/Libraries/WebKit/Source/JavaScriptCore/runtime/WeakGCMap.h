/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#include "DeferGC.h"
#include "Weak.h"
#include "WeakGCHashTable.h"
#include <wtf/HashMap.h>

namespace JSC {

// A UncheckedKeyHashMap with Weak<JSCell> values, which automatically removes values once they're garbage collected.

template<typename KeyArg, typename ValueArg, typename HashArg = DefaultHash<KeyArg>, typename KeyTraitsArg = HashTraits<KeyArg>>
class WeakGCMap final : public WeakGCHashTable {
    WTF_MAKE_FAST_ALLOCATED;
    typedef Weak<ValueArg> ValueType;
    typedef UncheckedKeyHashMap<KeyArg, ValueType, HashArg, KeyTraitsArg> HashMapType;

public:
    typedef typename HashMapType::KeyType KeyType;
    typedef typename HashMapType::AddResult AddResult;
    typedef typename HashMapType::iterator iterator;
    typedef typename HashMapType::const_iterator const_iterator;

    explicit WeakGCMap(VM&);
    ~WeakGCMap() final;

    ValueArg* get(const KeyType& key) const
    {
        return m_map.get(key);
    }

    AddResult set(const KeyType& key, ValueType value)
    {
        return m_map.set(key, WTFMove(value));
    }

    template<typename Functor>
    ValueArg* ensureValue(const KeyType& key, Functor&& functor)
    {
        // If functor invokes GC, GC can prune WeakGCMap, and manipulate UncheckedKeyHashMap while we are touching it in ensure function.
        // The functor must not invoke GC.
        DisallowGC disallowGC;
        AddResult result = m_map.ensure(key, std::forward<Functor>(functor));
        ValueArg* value = result.iterator->value.get();
        if (!result.isNewEntry && !value) {
            value = functor();
            result.iterator->value = WTFMove(value);
        }
        return value;
    }

    bool remove(const KeyType& key)
    {
        return m_map.remove(key);
    }

    void clear()
    {
        m_map.clear();
    }

    bool isEmpty() const
    {
        const_iterator it = m_map.begin();
        const_iterator end = m_map.end();
        while (it != end) {
            if (it->value)
                return true;
        }
        return false;
    }

    inline iterator find(const KeyType& key);

    inline const_iterator find(const KeyType& key) const;

    inline bool contains(const KeyType& key) const;

    void pruneStaleEntries() final;

    template<typename Func>
    void forEach(Func);

private:
    HashMapType m_map;
    VM& m_vm;
};

} // namespace JSC
