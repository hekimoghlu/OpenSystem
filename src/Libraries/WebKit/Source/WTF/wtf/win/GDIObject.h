/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#ifndef GDIObject_h
#define GDIObject_h

#include <algorithm>
#include <cstddef>
#include <memory>
#include <windows.h>
#include <wtf/Assertions.h>
#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename T> void deleteObject(T);

template<typename T> class GDIObject {
    WTF_MAKE_NONCOPYABLE(GDIObject);
    WTF_MAKE_FAST_ALLOCATED;
public:
    GDIObject() : m_object(0) { }
    GDIObject(std::nullptr_t) : m_object(0) { }
    ~GDIObject() { deleteObject<T>(m_object); }

    T get() const { return m_object; }

    void clear();
    T leak() WARN_UNUSED_RETURN;

    bool operator!() const { return !m_object; }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef const void* UnspecifiedBoolType;
    operator UnspecifiedBoolType() const { return m_object ? reinterpret_cast<UnspecifiedBoolType>(&m_object) : 0; }

    GDIObject<T>& operator=(std::nullptr_t) { clear(); return *this; }

    GDIObject(GDIObject&&);
    template<typename U> GDIObject(GDIObject<U>&&);

    GDIObject& operator=(GDIObject&&);
    template<typename U> GDIObject& operator=(GDIObject<U>&&);

    void swap(GDIObject& o) { std::swap(m_object, o.m_object); }

private:
    template<typename U> friend GDIObject<U> adoptGDIObject(U);
    GDIObject(T object) : m_object(object) { }

    GDIObject<T>& operator=(T);

    T m_object;
};

template<typename T> inline void GDIObject<T>::clear()
{
    T object = m_object;
    m_object = 0;
    deleteObject(object);
}

template<typename T> inline T GDIObject<T>::leak()
{
    T object = m_object;
    m_object = 0;
    return object;
}

template<typename T> inline GDIObject<T>::GDIObject(GDIObject<T>&& other)
    : m_object(other.leak())
{
}

template<typename T> inline GDIObject<T>& GDIObject<T>::operator=(GDIObject<T>&& other)
{
    auto object = WTFMove(other);
    swap(object);
    return *this;
}

template<typename T> inline GDIObject<T> adoptGDIObject(T object)
{
    return GDIObject<T>(object);
}

template<typename T> inline void swap(GDIObject<T>& a, GDIObject<T>& b)
{
    a.swap(b);
}

// Nearly all GDI types use the same DeleteObject call.
template<typename T> inline void deleteObject(T object)
{
    if (object)
        ::DeleteObject(object);
}

template<> inline void deleteObject<HDC>(HDC hdc)
{
    if (hdc)
        ::DeleteDC(hdc);
}

} // namespace WTF

using WTF::GDIObject;
using WTF::adoptGDIObject;

#endif // GDIObject_h
