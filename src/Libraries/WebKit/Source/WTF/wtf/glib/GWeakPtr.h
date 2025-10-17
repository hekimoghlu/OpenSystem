/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#if USE(GLIB)

#include <glib-object.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template <typename T> class GWeakPtr {
    WTF_MAKE_NONCOPYABLE(GWeakPtr);
public:
    GWeakPtr() = default;

    explicit GWeakPtr(T* ptr)
        : m_ptr(ptr)
    {
        RELEASE_ASSERT(!ptr || G_IS_OBJECT(ptr));
        addWeakPtr();
    }

    GWeakPtr(GWeakPtr&& other)
    {
        reset(other.get());
        other.reset();
    }

    ~GWeakPtr()
    {
        removeWeakPtr();
    }

    T& operator*() const
    {
        ASSERT(m_ptr);
        return *m_ptr;
    }

    T* operator->() const
    {
        ASSERT(m_ptr);
        return m_ptr;
    }

    T* get() const
    {
        return m_ptr;
    }

    void reset(T* ptr = nullptr)
    {
        RELEASE_ASSERT(!ptr || G_IS_OBJECT(ptr));
        removeWeakPtr();
        m_ptr = ptr;
        addWeakPtr();
    }

    GWeakPtr& operator=(std::nullptr_t)
    {
        reset();
        return *this;
    }

    GWeakPtr& operator=(GWeakPtr&& other)
    {
        reset(other.get());
        other.reset();
        return *this;
    }

    bool operator!() const { return !m_ptr; }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* GWeakPtr::*UnspecifiedBoolType;
    operator UnspecifiedBoolType() const { return m_ptr ? &GWeakPtr::m_ptr : 0; }

private:
    void addWeakPtr()
    {
        if (m_ptr)
            g_object_add_weak_pointer(G_OBJECT(m_ptr), reinterpret_cast<void**>(&m_ptr));
    }

    void removeWeakPtr()
    {
        if (m_ptr)
            g_object_remove_weak_pointer(G_OBJECT(m_ptr), reinterpret_cast<void**>(&m_ptr));
    }

    T* m_ptr { nullptr };
};

} // namespace WTF

using WTF::GWeakPtr;

#endif // USE(GLIB)
