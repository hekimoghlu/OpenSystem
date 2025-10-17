/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

#include "GRefPtr.h"

#include <glib-object.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template <typename T> class GThreadSafeWeakPtr {
    WTF_MAKE_NONCOPYABLE(GThreadSafeWeakPtr);
public:
    GThreadSafeWeakPtr()
    {
        g_weak_ref_init(&m_ref, nullptr);
    }

    explicit GThreadSafeWeakPtr(T* ptr)
    {
        RELEASE_ASSERT(!ptr || G_IS_OBJECT(ptr));
        g_weak_ref_init(&m_ref, ptr);
    }

    GThreadSafeWeakPtr(GThreadSafeWeakPtr&& other)
    {
        auto strongRef = other.get();
        g_weak_ref_set(&other.m_ref, nullptr);
        g_weak_ref_init(&m_ref, strongRef.get());
    }

    ~GThreadSafeWeakPtr()
    {
        g_weak_ref_clear(&m_ref);
    }

    WARN_UNUSED_RETURN GRefPtr<T> get()
    {
        return adoptGRef(reinterpret_cast<T*>(g_weak_ref_get(&m_ref)));
    }

    void reset(T* ptr = nullptr)
    {
        RELEASE_ASSERT(!ptr || G_IS_OBJECT(ptr));
        g_weak_ref_set(&m_ref, ptr);
    }

    GThreadSafeWeakPtr& operator=(std::nullptr_t)
    {
        reset();
        return *this;
    }

private:
    GWeakRef m_ref;
};

} // namespace WTF

using WTF::GThreadSafeWeakPtr;

#endif // USE(GLIB)
