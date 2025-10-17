/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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

#ifdef __cplusplus

#include <WebKit/WKType.h>
#include <algorithm>
#include <wtf/HashTraits.h>

namespace WebKit {

template<typename T> class WKRetainPtr {
public:
    using PtrType = T;
    using ValueType = std::remove_pointer_t<PtrType>;

    WKRetainPtr()
        : m_ptr(0)
    {
    }

    WKRetainPtr(PtrType ptr)
        : m_ptr(ptr)
    {
        if (ptr)
            WKRetain(ptr);
    }

    template<typename U> WKRetainPtr(const WKRetainPtr<U>& o)
        : m_ptr(o.get())
    {
        if (PtrType ptr = m_ptr)
            WKRetain(ptr);
    }
    
    WKRetainPtr(const WKRetainPtr& o)
        : m_ptr(o.m_ptr)
    {
        if (PtrType ptr = m_ptr)
            WKRetain(ptr);
    }

    template<typename U> WKRetainPtr(WKRetainPtr<U>&& o)
        : m_ptr(o.leakRef())
    {
    }
    
    WKRetainPtr(WKRetainPtr&& o)
        : m_ptr(o.leakRef())
    {
    }

    ~WKRetainPtr()
    {
        if (PtrType ptr = m_ptr)
            WKRelease(ptr);
    }

    // Hash table deleted values, which are only constructed and never copied or destroyed.
    WKRetainPtr(WTF::HashTableDeletedValueType)
        : m_ptr(hashTableDeletedValue())
    {
    }

    bool isHashTableDeletedValue() const { return m_ptr == hashTableDeletedValue(); }
    constexpr static T hashTableDeletedValue() { return reinterpret_cast<T>(-1); }

    PtrType get() const { return m_ptr; }

    void clear()
    {
        PtrType ptr = m_ptr;
        m_ptr = 0;
        if (ptr)
            WKRelease(ptr);
    }

    PtrType leakRef()
    {
        PtrType ptr = m_ptr;
        m_ptr = 0;
        return ptr;
    }
    
    PtrType operator->() const { return m_ptr; }
    bool operator!() const { return !m_ptr; }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef PtrType WKRetainPtr::*UnspecifiedBoolType;
    operator UnspecifiedBoolType() const { return m_ptr ? &WKRetainPtr::m_ptr : 0; }

    WKRetainPtr& operator=(const WKRetainPtr&);
    template<typename U> WKRetainPtr& operator=(const WKRetainPtr<U>&);
    WKRetainPtr& operator=(PtrType);
    template<typename U> WKRetainPtr& operator=(U*);

    WKRetainPtr& operator=(WKRetainPtr&&);
    template<typename U> WKRetainPtr& operator=(WKRetainPtr<U>&&);

    void adopt(PtrType);
    void swap(WKRetainPtr&);

private:
    template<typename U> friend WKRetainPtr<U> adoptWK(U);
    enum WKAdoptTag { AdoptWK };
    WKRetainPtr(WKAdoptTag, PtrType ptr)
        : m_ptr(ptr) { }

    PtrType m_ptr;
};

template<typename T> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(const WKRetainPtr<T>& o)
{
    PtrType optr = o.get();
    if (optr)
        WKRetain(optr);
    PtrType ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        WKRelease(ptr);
    return *this;
}

template<typename T> template<typename U> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(const WKRetainPtr<U>& o)
{
    PtrType optr = o.get();
    if (optr)
        WKRetain(optr);
    PtrType ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        WKRelease(ptr);
    return *this;
}

template<typename T> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(PtrType optr)
{
    if (optr)
        WKRetain(optr);
    PtrType ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        WKRelease(ptr);
    return *this;
}

template<typename T> template<typename U> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(U* optr)
{
    if (optr)
        WKRetain(optr);
    PtrType ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        WKRelease(ptr);
    return *this;
}

template<typename T> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(WKRetainPtr<T>&& o)
{
    adopt(o.leakRef());
    return *this;
}

template<typename T> template<typename U> inline WKRetainPtr<T>& WKRetainPtr<T>::operator=(WKRetainPtr<U>&& o)
{
    adopt(o.leakRef());
    return *this;
}

template<typename T> inline void WKRetainPtr<T>::adopt(PtrType optr)
{
    PtrType ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        WKRelease(ptr);
}

template<typename T> inline void WKRetainPtr<T>::swap(WKRetainPtr<T>& o)
{
    std::swap(m_ptr, o.m_ptr);
}

template<typename T> inline void swap(WKRetainPtr<T>& a, WKRetainPtr<T>& b)
{
    a.swap(b);
}

template<typename T, typename U> inline bool operator==(const WKRetainPtr<T>& a, const WKRetainPtr<U>& b)
{ 
    return a.get() == b.get(); 
}

template<typename T, typename U> inline bool operator==(const WKRetainPtr<T>& a, U* b)
{ 
    return a.get() == b; 
}

template<typename T, typename U> inline bool operator==(T* a, const WKRetainPtr<U>& b) 
{
    return a == b.get(); 
}

#if (defined(WIN32) || defined(_WIN32)) && !((_MSC_VER > 1900) && __clang__)
template<typename T> inline WKRetainPtr<T> adoptWK(T) _Check_return_;
#else
template<typename T> inline WKRetainPtr<T> adoptWK(T) __attribute__((warn_unused_result));
#endif
template<typename T> inline WKRetainPtr<T> adoptWK(T o)
{
    return WKRetainPtr<T>(WKRetainPtr<T>::AdoptWK, o);
}

template<typename T> inline WKRetainPtr<T> retainWK(T ptr)
{
    return ptr;
}

} // namespace WebKit

using WebKit::WKRetainPtr;
using WebKit::adoptWK;
using WebKit::retainWK;

namespace WTF {

template<typename> struct IsSmartPtr;
template<typename> struct DefaultHash;

template<typename T> struct IsSmartPtr<WKRetainPtr<T>> {
    WTF_INTERNAL static const bool value = true;
    WTF_INTERNAL static constexpr bool isNullable = true;
};

template<typename P> struct DefaultHash<WKRetainPtr<P>> : PtrHash<WKRetainPtr<P>> { };

template<typename P> struct HashTraits<WKRetainPtr<P>> : SimpleClassHashTraits<WKRetainPtr<P>> {
    static P emptyValue() { return nullptr; }
    static bool isEmptyValue(const WKRetainPtr<P>& value) { return !value; }

    typedef P PeekType;
    static PeekType peek(const WKRetainPtr<P>& value) { return value.get(); }
    static PeekType peek(P value) { return value; }
};

} // namespace WTF

#endif
