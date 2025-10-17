/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

#include <objc/runtime.h>
#include <type_traits>
#include <wtf/RetainPtr.h>
#include <wtf/spi/cocoa/objcSPI.h>

// Because ARC enablement is a compile-time choice, and we compile this header
// both ways, we need a separate copy of our code when ARC is enabled.
#if __has_feature(objc_arc)
#define WeakObjCPtr WeakObjCPtrArc
#endif

namespace WTF {

template<typename T> class WeakObjCPtr {
public:
    using ValueType = typename std::remove_pointer<T>::type;

    WeakObjCPtr() = default;

    WeakObjCPtr(ValueType *ptr)
#if __has_feature(objc_arc)
        : m_weakReference(ptr)
#endif
    {
#if !__has_feature(objc_arc)
        objc_initWeak(&m_weakReference, ptr);
#endif
    }

#if !__has_feature(objc_arc)
    WeakObjCPtr(const WeakObjCPtr& other)
    {
        objc_copyWeak(&m_weakReference, &other.m_weakReference);
    }

    WeakObjCPtr(WeakObjCPtr&& other)
    {
        objc_moveWeak(&m_weakReference, &other.m_weakReference);
    }

    ~WeakObjCPtr()
    {
        objc_destroyWeak(&m_weakReference);
    }
#endif

    WeakObjCPtr& operator=(ValueType *ptr)
    {
#if __has_feature(objc_arc)
        m_weakReference = ptr;
#else
        objc_storeWeak(&m_weakReference, (id)ptr);
#endif

        return *this;
    }

    bool operator!() const { return !get(); }
    explicit operator bool() const { return !!get(); }

    RetainPtr<ValueType> get() const;

    ValueType *getAutoreleased() const
    {
#if __has_feature(objc_arc)
        return static_cast<ValueType *>(m_weakReference);
#else
        return static_cast<ValueType *>(objc_loadWeak(&m_weakReference));
#endif

    }

    explicit operator ValueType *() const { return getAutoreleased(); }

private:
#if __has_feature(objc_arc)
    mutable __weak id m_weakReference { nullptr };
#else
    mutable id m_weakReference { nullptr };
#endif
};

template<typename T> WeakObjCPtr(T) -> WeakObjCPtr<std::remove_pointer_t<T>>;

#ifdef __OBJC__
template<typename T>
RetainPtr<typename WeakObjCPtr<T>::ValueType> WeakObjCPtr<T>::get() const
{
#if __has_feature(objc_arc)
    return static_cast<typename WeakObjCPtr<T>::ValueType *>(m_weakReference);
#else
    return adoptNS(objc_loadWeakRetained(&m_weakReference));
#endif
}
#endif

} // namespace WTF

using WTF::WeakObjCPtr;
