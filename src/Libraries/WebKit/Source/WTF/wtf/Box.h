/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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

#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {

// Box<T> is a reference-counted pointer to T that allocates T using FastMalloc and prepends a reference
// count to it.
template<typename T>
class Box {
    WTF_MAKE_FAST_ALLOCATED;
public:
    Box() = default;
    Box(Box&&) = default;
    Box(const Box&) = default;

    Box(std::nullptr_t)
    {
    }

    Box& operator=(Box&&) = default;
    Box& operator=(const Box&) = default;

    template<typename... Arguments>
    static Box create(Arguments&&... arguments)
    {
        Box result;
        result.m_data = adoptRef(new Data(std::forward<Arguments>(arguments)...));
        return result;
    }

    T* get() const { return &m_data->value; }

    T& operator*() const { return m_data->value; }
    T* operator->() const { return &m_data->value; }

    explicit operator bool() const { return static_cast<bool>(m_data); }
    
private:
    struct Data : ThreadSafeRefCounted<Data> {
        template<typename... Arguments>
        Data(Arguments&&... arguments)
            : value(std::forward<Arguments>(arguments)...)
        {
        }
        
        T value;
    };

    RefPtr<Data> m_data;
};

} // namespace WTF

using WTF::Box;
