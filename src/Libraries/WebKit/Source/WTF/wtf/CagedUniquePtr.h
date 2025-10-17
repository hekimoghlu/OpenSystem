/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include <wtf/CagedPtr.h>

namespace WTF {

template<Gigacage::Kind kind, typename T>
class CagedUniquePtr : public CagedPtr<kind, T> {
    static_assert(std::is_trivially_destructible<T>::value, "We expect the contents of a caged pointer to be trivially destructable.");
public:
    using Base = CagedPtr<kind, T>;
    CagedUniquePtr() = default;

    CagedUniquePtr(T* ptr)
        : Base(ptr)
    { }

    CagedUniquePtr(CagedUniquePtr&& ptr)
        : Base(std::forward<CagedUniquePtr&&>(ptr))
    { }
    
    CagedUniquePtr(const CagedUniquePtr&) = delete;
    
    template<typename... Arguments>
    static CagedUniquePtr create(size_t length, Arguments&&... arguments)
    {
        T* result = static_cast<T*>(Gigacage::malloc(kind, sizeof(T) * length));
        while (length--)
            new (result + length) T(arguments...);
        return CagedUniquePtr(result);
    }

    template<typename... Arguments>
    static CagedUniquePtr tryCreate(size_t length, Arguments&&... arguments)
    {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        T* result = static_cast<T*>(Gigacage::tryMalloc(kind, sizeof(T) * length));
        if (!result)
            return { };
        while (length--)
            new (result + length) T(arguments...);
        return CagedUniquePtr(result);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

    CagedUniquePtr& operator=(CagedUniquePtr&& ptr)
    {
        destroy();
        this->m_ptr = ptr.m_ptr;
        ptr.m_ptr = nullptr;
        return *this;
    }
    
    CagedUniquePtr& operator=(const CagedUniquePtr&) = delete;
    
    ~CagedUniquePtr()
    {
        destroy();
    }

private:
    void destroy()
    {
        T* ptr = Base::getUnsafe();
        if (!ptr)
            return;
        ptr->~T();
        Gigacage::free(kind, ptr);
    }
};

} // namespace WTF

using WTF::CagedUniquePtr;

