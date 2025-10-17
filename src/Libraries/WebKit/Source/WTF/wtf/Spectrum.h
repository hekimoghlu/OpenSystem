/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include <wtf/HashMap.h>
#include <wtf/Vector.h>
#include <algorithm>

namespace WTF {

template<typename T, typename CounterType = unsigned>
class Spectrum {
    WTF_MAKE_FAST_ALLOCATED;
public:
    typedef typename UncheckedKeyHashMap<T, CounterType>::iterator iterator;
    typedef typename UncheckedKeyHashMap<T, CounterType>::const_iterator const_iterator;
    
    Spectrum() { }
    
    void add(const T& key, CounterType count = 1)
    {
        Locker locker(m_lock);
        if (!count)
            return;
        typename UncheckedKeyHashMap<T, CounterType>::AddResult result = m_map.add(key, count);
        if (!result.isNewEntry)
            result.iterator->value += count;
    }
    
    template<typename U>
    void addAll(const Spectrum<T, U>& otherSpectrum)
    {
        for (auto& entry : otherSpectrum)
            add(entry.key, entry.count);
    }
    
    CounterType get(const T& key) const
    {
        Locker locker(m_lock);
        const_iterator iter = m_map.find(key);
        if (iter == m_map.end())
            return 0;
        return iter->value;
    }
    
    size_t size() const { return m_map.size(); }
    
    const_iterator begin() const { return m_map.begin(); }
    const_iterator end() const { return m_map.end(); }
    
    struct KeyAndCount {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        KeyAndCount() { }
        
        KeyAndCount(const T& key, CounterType count)
            : key(&key)
            , count(count)
        {
        }
        
        bool operator<(const KeyAndCount& other) const
        {
            if (count != other.count)
                return count < other.count;
            // This causes lower-ordered keys being returned first; this is really just
            // here to make sure that the order is somewhat deterministic rather than being
            // determined by hashing.
            return *key > *other.key;
        }

        const T* key;
        CounterType count;
    };

    Lock& getLock() { return m_lock; }
    
    // Returns a list ordered from lowest-count to highest-count.
    Vector<KeyAndCount> buildList(AbstractLocker&) const
    {
        Vector<KeyAndCount> list;
        for (const auto& entry : *this)
            list.append(KeyAndCount(entry.key, entry.value));
        
        std::sort(list.begin(), list.end());
        return list;
    }
    
    void clear()
    {
        Locker locker(m_lock);
        m_map.clear();
    }
    
    template<typename Functor>
    void removeIf(const Functor& functor)
    {
        Locker locker(m_lock);
        m_map.removeIf([&functor] (typename UncheckedKeyHashMap<T, CounterType>::KeyValuePairType& pair) {
                return functor(KeyAndCount(pair.key, pair.value));
            });
    }
    
private:
    mutable Lock m_lock;
    UncheckedKeyHashMap<T, CounterType> m_map;
};

} // namespace WTF

using WTF::Spectrum;
