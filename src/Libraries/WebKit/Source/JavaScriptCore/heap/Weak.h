/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#include "JSExportMacros.h"
#include <cstddef>
#include <wtf/Atomics.h>
#include <wtf/HashTraits.h>
#include <wtf/Noncopyable.h>
#include <wtf/VectorTraits.h>

namespace JSC {

class WeakImpl;
class WeakHandleOwner;
class VM;

template<typename T> class Weak {
    WTF_MAKE_NONCOPYABLE(Weak);
public:
    Weak() = default;

    constexpr Weak(std::nullptr_t)
        : Weak()
    {
    }

    Weak(T*, WeakHandleOwner* = nullptr, void* context = nullptr);
    Weak(VM&, T* t, WeakHandleOwner* owner = nullptr, void* context = nullptr)
        : Weak(t, owner, context)
    {
    }

    bool isHashTableDeletedValue() const;
    Weak(WTF::HashTableDeletedValueType);
    constexpr bool isHashTableEmptyValue() const { return !impl(); }

    Weak(Weak&&);

    ~Weak()
    {
        clear();
    }

    inline void swap(Weak&);

    inline Weak& operator=(Weak&&);

    inline bool operator!() const;
    inline T* operator->() const;
    inline T& operator*() const;
    inline T* get() const;

    inline void set(VM&, T*);

    inline bool was(T*) const;

    inline explicit operator bool() const;

    inline WeakImpl* leakImpl() WARN_UNUSED_RETURN;
    WeakImpl* unsafeImpl() const { return impl(); }
    void clear();

private:
    static inline WeakImpl* hashTableDeletedValue();
    WeakImpl* impl() const { return m_impl; }

    WeakImpl* m_impl { nullptr };
};

} // namespace JSC

namespace WTF {

template<typename T> struct VectorTraits<JSC::Weak<T>> : SimpleClassVectorTraits {
    static constexpr bool canCompareWithMemcmp = false;
};

template<typename T> struct HashTraits<JSC::Weak<T>> : SimpleClassHashTraits<JSC::Weak<T>> {
    typedef JSC::Weak<T> StorageType;

    typedef std::nullptr_t EmptyValueType;
    static EmptyValueType emptyValue() { return nullptr; }

    static constexpr bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const JSC::Weak<T>& value) { return value.isHashTableEmptyValue(); }

    typedef T* PeekType;
    static PeekType peek(const StorageType& value) { return value.get(); }
    static PeekType peek(EmptyValueType) { return PeekType(); }
};

} // namespace WTF
