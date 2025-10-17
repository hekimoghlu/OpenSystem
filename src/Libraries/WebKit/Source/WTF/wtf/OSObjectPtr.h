/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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

#include <os/object.h>
#include <wtf/StdLibExtras.h>

// Because ARC enablement is a compile-time choice, and we compile this header
// both ways, we need a separate copy of our code when ARC is enabled.
#if __has_feature(objc_arc)
#define adoptOSObject adoptOSObjectArc
#define retainOSObject retainOSObjectArc
#define releaseOSObject releaseOSObjectArc
#endif

namespace WTF {

template<typename> class OSObjectPtr;
template<typename T> OSObjectPtr<T> adoptOSObject(T) WARN_UNUSED_RETURN;

template<typename T> static inline void retainOSObject(T ptr)
{
#if __has_feature(objc_arc)
    UNUSED_PARAM(ptr);
#else
    os_retain(ptr);
#endif
}

template<typename T> static inline void releaseOSObject(T ptr)
{
#if __has_feature(objc_arc)
    UNUSED_PARAM(ptr);
#else
    os_release(ptr);
#endif
}

template<typename T> class OSObjectPtr {
public:
    OSObjectPtr()
        : m_ptr(nullptr)
    {
    }

    ~OSObjectPtr()
    {
        if (m_ptr)
            releaseOSObject(m_ptr);
    }

    T get() const { return m_ptr; }

    explicit operator bool() const { return m_ptr; }
    bool operator!() const { return !m_ptr; }

    OSObjectPtr(const OSObjectPtr& other)
        : m_ptr(other.m_ptr)
    {
        if (m_ptr)
            retainOSObject(m_ptr);
    }

    OSObjectPtr(OSObjectPtr&& other)
        : m_ptr(WTFMove(other.m_ptr))
    {
        other.m_ptr = nullptr;
    }

    OSObjectPtr(T ptr)
        : m_ptr(WTFMove(ptr))
    {
        if (m_ptr)
            retainOSObject(m_ptr);
    }

    OSObjectPtr& operator=(const OSObjectPtr& other)
    {
        OSObjectPtr ptr = other;
        swap(ptr);
        return *this;
    }

    OSObjectPtr& operator=(OSObjectPtr&& other)
    {
        OSObjectPtr ptr = WTFMove(other);
        swap(ptr);
        return *this;
    }

    OSObjectPtr& operator=(std::nullptr_t)
    {
        if (m_ptr)
            releaseOSObject(m_ptr);
        m_ptr = nullptr;
        return *this;
    }

    OSObjectPtr& operator=(T other)
    {
        OSObjectPtr ptr = WTFMove(other);
        swap(ptr);
        return *this;
    }

    void swap(OSObjectPtr& other)
    {
        std::swap(m_ptr, other.m_ptr);
    }

    T leakRef() WARN_UNUSED_RETURN
    {
        return std::exchange(m_ptr, nullptr);
    }

    friend OSObjectPtr adoptOSObject<T>(T) WARN_UNUSED_RETURN;

private:
    struct AdoptOSObject { };
    OSObjectPtr(AdoptOSObject, T ptr)
        : m_ptr(WTFMove(ptr))
    {
    }

    T m_ptr;
};

template<typename T> inline OSObjectPtr<T> adoptOSObject(T ptr)
{
    return OSObjectPtr<T> { typename OSObjectPtr<T>::AdoptOSObject { }, WTFMove(ptr) };
}

} // namespace WTF

using WTF::OSObjectPtr;
using WTF::adoptOSObject;
