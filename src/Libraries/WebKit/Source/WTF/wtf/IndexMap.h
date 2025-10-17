/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

#include <wtf/IndexKeyType.h>
#include <wtf/Vector.h>

namespace WTF {

// This is a map for keys that have an index(). It's super efficient for BasicBlocks. It's only
// efficient for Values if you don't create too many of these maps, since Values can have very
// sparse indices and there are a lot of Values.

template<typename Key, typename Value>
class IndexMap {
    WTF_MAKE_FAST_ALLOCATED;
public:
    IndexMap() = default;
    IndexMap(IndexMap&&) = default;
    IndexMap& operator=(IndexMap&&) = default;
    IndexMap(const IndexMap&) = default;
    IndexMap& operator=(const IndexMap&) = default;
    
    template<typename... Args>
    explicit IndexMap(size_t size, Args&&... args)
        : m_vector(size, Value(std::forward<Args>(args)...))
    {
    }

    template<typename... Args>
    void resize(size_t size, Args&&... args)
    {
        m_vector.fill(Value(std::forward<Args>(args)...), size);
    }

    template<typename... Args>
    void clear(Args&&... args)
    {
        m_vector.fill(Value(std::forward<Args>(args)...), m_vector.size());
    }

    size_t size() const { return m_vector.size(); }

    Value& at(const Key& key)
    {
        return m_vector[IndexKeyType<Key>::index(key)];
    }
    
    const Value& at(const Key& key) const
    {
        return m_vector[IndexKeyType<Key>::index(key)];
    }

    Value& at(size_t index)
    {
        return m_vector[index];
    }

    const Value& at(size_t index) const
    {
        return m_vector[index];
    }
    
    Value& operator[](size_t index) { return at(index); }
    const Value& operator[](size_t index) const { return at(index); }
    Value& operator[](const Key& key) { return at(key); }
    const Value& operator[](const Key& key) const { return at(key); }
    
    template<typename PassedValue>
    void append(const Key& key, PassedValue&& value)
    {
        RELEASE_ASSERT(IndexKeyType<Key>::index(key) == m_vector.size());
        m_vector.append(std::forward<PassedValue>(value));
    }

private:
    Vector<Value, 0, UnsafeVectorOverflow> m_vector;
};

} // namespace WTF

using WTF::IndexMap;
