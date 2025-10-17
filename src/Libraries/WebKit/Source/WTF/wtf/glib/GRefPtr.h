/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#include <algorithm>
#include <glib.h>
#include <wtf/HashTraits.h>

extern "C" {
    typedef struct _GDBusConnection GDBusConnection;
    typedef struct _GDBusNodeInfo GDBusNodeInfo;

    GDBusNodeInfo* g_dbus_node_info_ref(GDBusNodeInfo*);
    void g_dbus_node_info_unref(GDBusNodeInfo*);

    // Since GLib 2.56 a g_object_ref_sink() macro may be defined which propagates
    // the type of the parameter to the returned value, but it conflicts with the
    // declaration below, causing an error when glib-object.h is included before
    // this file. Thus, add the forward declarations only when the macro is not
    // present.
#ifndef g_object_ref_sink
    void g_object_unref(gpointer);
    gpointer g_object_ref_sink(gpointer);
#endif
};

namespace WTF {

enum GRefPtrAdoptType { GRefPtrAdopt };
template <typename T> inline T* refGPtr(T*);
template <typename T> inline void derefGPtr(T*);
template <typename T> class GRefPtr;
template <typename T> GRefPtr<T> adoptGRef(T*);

template <typename T> class GRefPtr {
public:
    typedef T ValueType;
    typedef ValueType* PtrType;

    GRefPtr() : m_ptr(0) { }

    GRefPtr(T* ptr)
        : m_ptr(ptr)
    {
        if (ptr)
            refGPtr(ptr);
    }

    GRefPtr(const GRefPtr& o)
        : m_ptr(o.m_ptr)
    {
        if (T* ptr = m_ptr)
            refGPtr(ptr);
    }

    template <typename U> GRefPtr(const GRefPtr<U>& o)
        : m_ptr(o.get())
    {
        if (T* ptr = m_ptr)
            refGPtr(ptr);
    }

    GRefPtr(GRefPtr&& o) : m_ptr(o.leakRef()) { }
    template <typename U> GRefPtr(GRefPtr<U>&& o) : m_ptr(o.leakRef()) { }

    ~GRefPtr()
    {
        if (T* ptr = m_ptr)
            derefGPtr(ptr);
    }

    void clear()
    {
        T* ptr = m_ptr;
        m_ptr = 0;
        if (ptr)
            derefGPtr(ptr);
    }

    T* leakRef() WARN_UNUSED_RETURN
    {
        T* ptr = m_ptr;
        m_ptr = 0;
        return ptr;
    }

    T*& outPtr()
    {
        clear();
        return m_ptr;
    }

    // Hash table deleted values, which are only constructed and never copied or destroyed.
    GRefPtr(HashTableDeletedValueType) : m_ptr(hashTableDeletedValue()) { }
    bool isHashTableDeletedValue() const { return m_ptr == hashTableDeletedValue(); }

    T* get() const { return m_ptr; }
    T& operator*() const { return *m_ptr; }
    ALWAYS_INLINE T* operator->() const { return m_ptr; }

    bool operator!() const { return !m_ptr; }

    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* GRefPtr::*UnspecifiedBoolType;
    operator UnspecifiedBoolType() const { return m_ptr ? &GRefPtr::m_ptr : 0; }

    GRefPtr& operator=(const GRefPtr&);
    GRefPtr& operator=(GRefPtr&&);
    GRefPtr& operator=(T*);
    template <typename U> GRefPtr& operator=(const GRefPtr<U>&);

    void swap(GRefPtr&);
    friend GRefPtr adoptGRef<T>(T*);

private:
    static T* hashTableDeletedValue() { return reinterpret_cast<T*>(-1); }
    // Adopting constructor.
    GRefPtr(T* ptr, GRefPtrAdoptType) : m_ptr(ptr) {}

    T* m_ptr;
};

template <typename T> inline GRefPtr<T>& GRefPtr<T>::operator=(const GRefPtr<T>& o)
{
    T* optr = o.get();
    if (optr)
        refGPtr(optr);
    T* ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        derefGPtr(ptr);
    return *this;
}

template <typename T> inline GRefPtr<T>& GRefPtr<T>::operator=(GRefPtr<T>&& o)
{
    GRefPtr ptr = WTFMove(o);
    swap(ptr);
    return *this;
}

template <typename T> inline GRefPtr<T>& GRefPtr<T>::operator=(T* optr)
{
    T* ptr = m_ptr;
    if (optr)
        refGPtr(optr);
    m_ptr = optr;
    if (ptr)
        derefGPtr(ptr);
    return *this;
}

template <class T> inline void GRefPtr<T>::swap(GRefPtr<T>& o)
{
    std::swap(m_ptr, o.m_ptr);
}

template <class T> inline void swap(GRefPtr<T>& a, GRefPtr<T>& b)
{
    a.swap(b);
}

template <typename T, typename U> inline bool operator==(const GRefPtr<T>& a, const GRefPtr<U>& b)
{
    return a.get() == b.get();
}

template <typename T, typename U> inline bool operator==(const GRefPtr<T>& a, U* b)
{
    return a.get() == b;
}

template <typename T, typename U> inline bool operator==(T* a, const GRefPtr<U>& b)
{
    return a == b.get();
}

template <typename T, typename U> inline bool operator!=(const GRefPtr<T>& a, const GRefPtr<U>& b)
{
    return a.get() != b.get();
}

template <typename T, typename U> inline bool operator!=(const GRefPtr<T>& a, U* b)
{
    return a.get() != b;
}

template <typename T, typename U> inline bool operator!=(T* a, const GRefPtr<U>& b)
{
    return a != b.get();
}

template <typename T, typename U> inline GRefPtr<T> static_pointer_cast(const GRefPtr<U>& p)
{
    return GRefPtr<T>(static_cast<T*>(p.get()));
}

template <typename T, typename U> inline GRefPtr<T> const_pointer_cast(const GRefPtr<U>& p)
{
    return GRefPtr<T>(const_cast<T*>(p.get()));
}

template <typename T> struct IsSmartPtr<GRefPtr<T>> {
    static const bool value = true;
    static constexpr bool isNullable = true;
};

template <typename T> GRefPtr<T> adoptGRef(T* p)
{
    return GRefPtr<T>(p, GRefPtrAdopt);
}

template <> WTF_EXPORT_PRIVATE GHashTable* refGPtr(GHashTable* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GHashTable* ptr);
template <> WTF_EXPORT_PRIVATE GMainContext* refGPtr(GMainContext* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GMainContext* ptr);
template <> WTF_EXPORT_PRIVATE GMainLoop* refGPtr(GMainLoop* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GMainLoop* ptr);
template <> WTF_EXPORT_PRIVATE GVariant* refGPtr(GVariant* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GVariant* ptr);
template <> WTF_EXPORT_PRIVATE GVariantBuilder* refGPtr(GVariantBuilder* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GVariantBuilder* ptr);
template <> WTF_EXPORT_PRIVATE GSource* refGPtr(GSource* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GSource* ptr);
template <> WTF_EXPORT_PRIVATE GPtrArray* refGPtr(GPtrArray*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GPtrArray*);
template <> WTF_EXPORT_PRIVATE GByteArray* refGPtr(GByteArray*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GByteArray*);
template <> WTF_EXPORT_PRIVATE GBytes* refGPtr(GBytes*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GBytes*);
template <> WTF_EXPORT_PRIVATE GClosure* refGPtr(GClosure*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GClosure*);
template <> WTF_EXPORT_PRIVATE GRegex* refGPtr(GRegex*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GRegex*);
template <> WTF_EXPORT_PRIVATE GMappedFile* refGPtr(GMappedFile*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GMappedFile*);
template <> WTF_EXPORT_PRIVATE GDateTime* refGPtr(GDateTime* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GDateTime* ptr);
template <> WTF_EXPORT_PRIVATE GDBusNodeInfo* refGPtr(GDBusNodeInfo* ptr);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GDBusNodeInfo* ptr);
template <> WTF_EXPORT_PRIVATE GArray* refGPtr(GArray*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GArray*);
template <> WTF_EXPORT_PRIVATE GResource* refGPtr(GResource*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GResource*);
template <> WTF_EXPORT_PRIVATE GUri* refGPtr(GUri*);
template <> WTF_EXPORT_PRIVATE void derefGPtr(GUri*);

template <typename T> inline T* refGPtr(T* ptr)
{
    if (ptr)
        g_object_ref_sink(ptr);
    return ptr;
}

template <typename T> inline void derefGPtr(T* ptr)
{
    if (ptr)
        g_object_unref(ptr);
}

template<typename P> struct DefaultHash<GRefPtr<P>> : PtrHash<GRefPtr<P>> { };

template<typename P> struct HashTraits<GRefPtr<P>> : SimpleClassHashTraits<GRefPtr<P>> {
    static P* emptyValue() { return nullptr; }

    typedef P* PeekType;
    static PeekType peek(const GRefPtr<P>& value) { return value.get(); }
    static PeekType peek(P* value) { return value; }

    static void customDeleteBucket(GRefPtr<P>& value)
    {
        // See unique_ptr's customDeleteBucket() for an explanation.
        ASSERT(!SimpleClassHashTraits<GRefPtr<P>>::isDeletedValue(value));
        auto valueToBeDestroyed = WTFMove(value);
        SimpleClassHashTraits<GRefPtr<P>>::constructDeletedValue(value);
    }
};

} // namespace WTF

using WTF::GRefPtr;
using WTF::adoptGRef;

#endif // USE(GLIB)
