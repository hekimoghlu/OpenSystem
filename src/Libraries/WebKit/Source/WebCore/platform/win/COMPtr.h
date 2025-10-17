/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#ifndef COMPtr_h
#define COMPtr_h

#include <unknwn.h>
#include <wtf/Assertions.h>
#include <wtf/HashTraits.h>

#ifdef __midl
typedef LONG HRESULT;
#else
typedef _Return_type_success_(return >= 0) long HRESULT;
#endif // __midl

// FIXME: Should we put this into the WebCore namespace and use "using" on it
// as we do with things in WTF? 

enum AdoptCOMTag { AdoptCOM };
enum QueryTag { Query };
enum CreateTag { Create };

template<typename T> class COMPtr {
public:
    using PtrType = T*;
    COMPtr() : m_ptr(nullptr) { }
    COMPtr(T* ptr) : m_ptr(ptr) { if (m_ptr) m_ptr->AddRef(); }
    COMPtr(AdoptCOMTag, T* ptr) : m_ptr(ptr) { }
    COMPtr(const COMPtr& o) : m_ptr(o.m_ptr) { if (T* ptr = m_ptr) ptr->AddRef(); }
    COMPtr(COMPtr&& o) : m_ptr(o.leakRef()) { }

    COMPtr(QueryTag, IUnknown* ptr) : m_ptr(copyQueryInterfaceRef(ptr)) { }
    template<typename U> COMPtr(QueryTag, const COMPtr<U>& ptr) : m_ptr(copyQueryInterfaceRef(ptr.get())) { }

    COMPtr(CreateTag, const IID& clsid) : m_ptr(createInstance(clsid)) { }

    // Hash table deleted values, which are only constructed and never copied or destroyed.
    COMPtr(WTF::HashTableDeletedValueType) : m_ptr(hashTableDeletedValue()) { }
    bool isHashTableDeletedValue() const { return m_ptr == hashTableDeletedValue(); }

    ~COMPtr() { if (m_ptr) m_ptr->Release(); }

    T* get() const { return m_ptr; }

    void clear();
    T* leakRef();

    T& operator*() const { return *m_ptr; }
    T* operator->() const { return m_ptr; }

    T** operator&() { ASSERT(!m_ptr); return &m_ptr; }

    bool operator!() const { return !m_ptr; }
    
    // This conversion operator allows implicit conversion to bool but not to other integer types.
    typedef T* (COMPtr::*UnspecifiedBoolType)() const;
    operator UnspecifiedBoolType() const { return m_ptr ? &COMPtr::get : 0; }

    COMPtr& operator=(const COMPtr&);
    COMPtr& operator=(COMPtr&&);
    COMPtr& operator=(T*);
    template<typename U> COMPtr& operator=(const COMPtr<U>&);

    void query(IUnknown* ptr) { adoptRef(copyQueryInterfaceRef(ptr)); }
    template<typename U> void query(const COMPtr<U>& ptr) { query(ptr.get()); }

    void create(const IID& clsid) { adoptRef(createInstance(clsid)); }

    template<typename U> HRESULT copyRefTo(U**);
    void adoptRef(T*);

private:
    static T* copyQueryInterfaceRef(IUnknown*);
    static T* createInstance(const IID& clsid);
    static T* hashTableDeletedValue() { return reinterpret_cast<T*>(-1); }

    T* m_ptr;
};

template<typename T> inline COMPtr<T> adoptCOM(T *ptr)
{
    return COMPtr<T>(AdoptCOM, ptr);
}

template<typename T> inline void COMPtr<T>::clear()
{
    if (T* ptr = m_ptr) {
        m_ptr = 0;
        ptr->Release();
    }
}

template<typename T> inline T* COMPtr<T>::leakRef()
{
    T* ptr = m_ptr;
    m_ptr = 0;
    return ptr;
}

template<typename T> inline T* COMPtr<T>::createInstance(const IID& clsid)
{
    T* result;
    if (FAILED(CoCreateInstance(clsid, 0, CLSCTX_ALL, __uuidof(result), reinterpret_cast<void**>(&result))))
        return 0;
    return result;
}

template<typename T> inline T* COMPtr<T>::copyQueryInterfaceRef(IUnknown* ptr)
{
    if (!ptr)
        return 0;
    T* result;
    if (FAILED(ptr->QueryInterface(&result)))
        return 0;
    return result;
}

template<typename T> template<typename U> inline HRESULT COMPtr<T>::copyRefTo(U** ptr)
{
    if (!ptr)
        return E_POINTER;
    *ptr = m_ptr;
    if (m_ptr)
        m_ptr->AddRef();
    return S_OK;
}

template<typename T> inline void COMPtr<T>::adoptRef(T *ptr)
{
    if (m_ptr)
        m_ptr->Release();
    m_ptr = ptr;
}

template<typename T> inline COMPtr<T>& COMPtr<T>::operator=(const COMPtr<T>& o)
{
    T* optr = o.get();
    if (optr)
        optr->AddRef();
    T* ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        ptr->Release();
    return *this;
}

template<typename T> inline COMPtr<T>& COMPtr<T>::operator=(COMPtr<T>&& o)
{
    if (T* ptr = m_ptr)
        ptr->Release();
    m_ptr = o.leakRef();
    return *this;
}

template<typename T> template<typename U> inline COMPtr<T>& COMPtr<T>::operator=(const COMPtr<U>& o)
{
    T* optr = o.get();
    if (optr)
        optr->AddRef();
    T* ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        ptr->Release();
    return *this;
}

template<typename T> inline COMPtr<T>& COMPtr<T>::operator=(T* optr)
{
    if (optr)
        optr->AddRef();
    T* ptr = m_ptr;
    m_ptr = optr;
    if (ptr)
        ptr->Release();
    return *this;
}

template<typename T, typename U> inline bool operator==(const COMPtr<T>& a, const COMPtr<U>& b)
{
    return a.get() == b.get();
}

template<typename T, typename U> inline bool operator==(const COMPtr<T>& a, U* b)
{
    return a.get() == b;
}

template<typename T, typename U> inline bool operator==(T* a, const COMPtr<U>& b) 
{
    return a == b.get();
}

template<typename T, typename U> inline bool operator!=(const COMPtr<T>& a, const COMPtr<U>& b)
{
    return a.get() != b.get();
}

template<typename T, typename U> inline bool operator!=(const COMPtr<T>& a, U* b)
{
    return a.get() != b;
}

template<typename T, typename U> inline bool operator!=(T* a, const COMPtr<U>& b)
{
    return a != b.get();
}

#if ASSERT_ENABLED
inline unsigned refCount(IUnknown* ptr)
{
    if (!ptr)
        return 0;

    unsigned temp = ptr->AddRef();
    unsigned value = ptr->Release();
    ASSERT(temp = value + 1);
    return value;
}
#endif

namespace WTF {

template<typename P> struct IsSmartPtr<COMPtr<P>> {
    static const bool value = true;
    static constexpr bool isNullable = true;
};

template<typename P> struct HashTraits<COMPtr<P> > : SimpleClassHashTraits<COMPtr<P>> {
    static P* emptyValue() { return nullptr; }

    typedef P* PeekType;
    static PeekType peek(const COMPtr<P>& value) { return value.get(); }
    static PeekType peek(P* value) { return value; }
};

template<typename P> struct DefaultHash<COMPtr<P>> : PtrHash<COMPtr<P>> { };

}

#endif
