/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#include <wtf/WeakHashMap.h>

namespace WTF {

template<typename Value, typename WeakPtrImpl = DefaultWeakPtrImpl>
class WeakHashCountedSet {
    WTF_MAKE_FAST_ALLOCATED;
private:
    using ImplType = WeakHashMap<Value, unsigned, WeakPtrImpl>;
public:
    using ValueType = Value;
    using iterator = typename ImplType::iterator;
    using const_iterator = typename ImplType::const_iterator;
    using AddResult = typename ImplType::AddResult;

    // Iterators iterate over pairs of values and counts.
    iterator begin() { return m_impl.begin(); }
    iterator end() { return m_impl.end(); }
    const_iterator begin() const { return m_impl.begin(); }
    const_iterator end() const { return m_impl.end(); }

    iterator find(const ValueType& value) { return m_impl.find(value); }
    const_iterator find(const ValueType& value) const { return m_impl.find(value); }
    bool contains(const ValueType& value) const { return m_impl.contains(value); }

    bool isEmptyIgnoringNullReferences() const { return m_impl.isEmptyIgnoringNullReferences(); }
    unsigned computeSize() const { return m_impl.computeSize(); }

    // Increments the count if an equal value is already present.
    // The return value includes both an iterator to the value's location,
    // and an isNewEntry bool that indicates whether it is a new or existing entry.
    AddResult add(const ValueType&);
    AddResult add(ValueType&&);

    // Decrements the count of the value, and removes it if count goes down to zero.
    // Returns true if the value is removed.
    bool remove(const ValueType&);
    bool remove(iterator);

    // Removes the value, regardless of its count.
    // Returns true if a value was removed.
    bool removeAll(const ValueType&);
    bool removeAll(iterator);

    // Clears the whole set.
    void clear() { m_impl.clear(); }

private:
    WeakHashMap<Value, unsigned, WeakPtrImpl> m_impl;
};

template<typename Value, typename WeakPtrImpl>
inline auto WeakHashCountedSet<Value, WeakPtrImpl>::add(const ValueType &value) -> AddResult
{
    auto result = m_impl.add(value, 0);
    ++result.iterator->value;
    return result;
}

template<typename Value, typename WeakPtrImpl>
inline auto WeakHashCountedSet<Value, WeakPtrImpl>::add(ValueType&& value) -> AddResult
{
    auto result = m_impl.add(std::forward<Value>(value), 0);
    ++result.iterator->value;
    return result;
}

template<typename Value, typename WeakPtrImpl>
inline bool WeakHashCountedSet<Value, WeakPtrImpl>::remove(const ValueType& value)
{
    return remove(find(value));
}

template<typename Value, typename WeakPtrImpl>
inline bool WeakHashCountedSet<Value, WeakPtrImpl>::remove(iterator it)
{
    if (it == end())
        return false;

    unsigned oldVal = it->value;
    ASSERT(oldVal);
    unsigned newVal = oldVal - 1;
    if (newVal) {
        it->value = newVal;
        return false;
    }

    m_impl.remove(it);
    return true;
}

template<typename Value, typename WeakPtrImpl>
inline bool WeakHashCountedSet<Value, WeakPtrImpl>::removeAll(const ValueType& value)
{
    return removeAll(find(value));
}

template<typename Value, typename WeakPtrImpl>
inline bool WeakHashCountedSet<Value, WeakPtrImpl>::removeAll(iterator it)
{
    if (it == end())
        return false;

    m_impl.remove(it);
    return true;
}

template<typename Value, typename WeakMapImpl>
size_t containerSize(const WeakHashCountedSet<Value, WeakMapImpl>& container) { return container.computeSize(); }

template<typename Value>
using SingleThreadWeakHashCountedSet = WeakHashCountedSet<Value, SingleThreadWeakPtrImpl>;

} // namespace WTF

using WTF::SingleThreadWeakHashCountedSet;
using WTF::WeakHashCountedSet;
