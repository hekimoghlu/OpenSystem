/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#include <memory>
#include <wtf/CompactPointerTuple.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename T, typename Type, typename Deleter = std::default_delete<T>> class CompactUniquePtrTuple;

template<typename T, typename Type, typename... Args>
ALWAYS_INLINE CompactUniquePtrTuple<T, Type> makeCompactUniquePtr(Args&&... args)
{
    return CompactUniquePtrTuple<T, Type>(makeUnique<T>(std::forward<Args>(args)...));
}

template<typename T, typename Type, typename Deleter, typename... Args>
ALWAYS_INLINE CompactUniquePtrTuple<T, Type, Deleter> makeCompactUniquePtr(Args&&... args)
{
    return CompactUniquePtrTuple<T, Type, Deleter>(makeUnique<T>(std::forward<Args>(args)...));
}

template<typename T, typename Type, typename Deleter>
class CompactUniquePtrTuple final {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(CompactUniquePtrTuple);

    static_assert(::allowCompactPointers<T>());
public:
    CompactUniquePtrTuple() = default;

    template <typename U, typename UDeleter, typename = std::enable_if_t<std::is_same<UDeleter, Deleter>::value || std::is_same<UDeleter, std::default_delete<U>>::value>>
    CompactUniquePtrTuple(CompactUniquePtrTuple<U, Type, UDeleter>&& other)
        : m_data { std::exchange(other.m_data, { }) }
    {
    }

    ~CompactUniquePtrTuple()
    {
        setPointer(nullptr);
    }

    template <typename U, typename UDeleter, typename = std::enable_if_t<std::is_same<UDeleter, Deleter>::value || std::is_same<UDeleter, std::default_delete<U>>::value>>
    CompactUniquePtrTuple<T, Type, Deleter>& operator=(CompactUniquePtrTuple<U, Type, UDeleter>&& other)
    {
        CompactUniquePtrTuple moved { WTFMove(other) };
        std::swap(m_data, moved.m_data);
        return *this;
    }

    T* pointer() const { return m_data.pointer(); }

    std::unique_ptr<T, Deleter> moveToUniquePtr()
    {
        T* pointer = m_data.pointer();
        m_data.setPointer(nullptr);
        return std::unique_ptr<T, Deleter>(pointer);
    }

    void setPointer(std::nullptr_t)
    {
        deletePointer();
        m_data.setPointer(nullptr);
    }

    template <typename U, typename UDeleter, typename = std::enable_if_t<std::is_same<UDeleter, Deleter>::value || std::is_same<UDeleter, std::default_delete<U>>::value>>
    void setPointer(std::unique_ptr<U, UDeleter>&& pointer)
    {
        deletePointer();
        m_data.setPointer(pointer.release());
    }

    Type type() const { return m_data.type(); }

    void setType(Type type)
    {
        m_data.setType(type);
    }

private:
    CompactUniquePtrTuple(std::unique_ptr<T>&& pointer)
    {
        m_data.setPointer(pointer.release());
    }

    void deletePointer()
    {
        if (T* pointer = m_data.pointer())
            Deleter()(pointer);
    }

    template<typename U, typename E, typename... Args> friend CompactUniquePtrTuple<U, E> makeCompactUniquePtr(Args&&... args);
    template<typename U, typename E, typename D, typename... Args> friend CompactUniquePtrTuple<U, E, D> makeCompactUniquePtr(Args&&... args);

    template <typename, typename, typename> friend class CompactUniquePtrTuple;

    CompactPointerTuple<T*, Type> m_data;
};

} // namespace WTF

using WTF::CompactUniquePtrTuple;
using WTF::makeCompactUniquePtr;
