/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include <wtf/HashTraits.h>
#include <wtf/Ref.h>

namespace WTF {

template <typename T>
class DataRef {
public:
    DataRef(Ref<T>&& data)
        : m_data(WTFMove(data))
    {
    }

    DataRef(const DataRef& other)
        : m_data(other.m_data.copyRef())
    {
    }

    DataRef& operator=(const DataRef& other)
    {
        m_data = other.m_data.copyRef();
        return *this;
    }

    DataRef(DataRef&&) = default;
    DataRef& operator=(DataRef&&) = default;

    DataRef replace(DataRef&& other)
    {
        return m_data.replace(WTFMove(other.m_data));
    }

    operator const T&() const
    {
        return m_data;
    }

    const T* ptr() const
    {
        return m_data.ptr();
    }

    const T& get() const
    {
        return m_data;
    }

    const T& operator*() const
    {
        return m_data;
    }

    const T* operator->() const
    {
        return m_data.ptr();
    }

    T& access()
    {
        if (!m_data->hasOneRef())
            m_data = m_data->copy();
        return m_data;
    }

    bool operator==(const DataRef& other) const
    {
        return m_data.ptr() == other.m_data.ptr() || m_data.get() == other.m_data.get();
    }

    DataRef(HashTableDeletedValueType)
        : m_data(HashTableDeletedValue)
    {
    }
    bool isHashTableDeletedValue() const { return m_data.isHashTableDeletedValue(); }

    DataRef(HashTableEmptyValueType)
        : m_data(HashTableEmptyValue)
    {
    }
    bool isHashTableEmptyValue() const { return m_data.isHashTableEmptyValue(); }
    static T* hashTableEmptyValue() { return nullptr; }

private:
    Ref<T> m_data;
};

template<typename T> struct HashTraits<DataRef<T>> : SimpleClassHashTraits<DataRef<T>> {
    static constexpr bool emptyValueIsZero = true;
    static DataRef<T> emptyValue() { return HashTableEmptyValue; }

    template <typename>
    static void constructEmptyValue(DataRef<T>& slot)
    {
        new (NotNull, std::addressof(slot)) DataRef<T>(HashTableEmptyValue);
    }

    static constexpr bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const DataRef<T>& value) { return value.isHashTableEmptyValue(); }
};

} // namespace WTF

using WTF::DataRef;
