/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/Ref.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename OwnerType, typename T>
class LazyRef {
    WTF_MAKE_NONCOPYABLE(LazyRef);
    WTF_MAKE_NONMOVABLE(LazyRef);
public:
    LazyRef() = default;

    template<typename Func>
    LazyRef(const Func& func)
    {
        initLater(func);
    }

    typedef T* (*FuncType)(OwnerType&, LazyRef&);

    ~LazyRef()
    {
        ASSERT(!(m_pointer & initializingTag));
        if (m_pointer & lazyTag)
            return;
        uintptr_t pointer = std::exchange(m_pointer, 0);
        if (pointer)
            std::bit_cast<T*>(pointer)->deref();
    }

    bool isInitialized() const { return !(m_pointer & lazyTag); }

    const T& get(const OwnerType& owner) const
    {
        return const_cast<LazyRef&>(*this).get(const_cast<OwnerType&>(owner));
    }

    T& get(OwnerType& owner)
    {
        ASSERT(m_pointer);
        ASSERT(!(m_pointer & initializingTag));
        if (UNLIKELY(m_pointer & lazyTag)) {
            FuncType func = *std::bit_cast<FuncType*>(m_pointer & ~(lazyTag | initializingTag));
            return *func(owner, *this);
        }
        return *std::bit_cast<T*>(m_pointer);
    }

    const T* getIfExists() const
    {
        return const_cast<LazyRef&>(*this).getIfExists();
    }

    T* getIfExists()
    {
        ASSERT(m_pointer);
        if (m_pointer & lazyTag)
            return nullptr;
        return std::bit_cast<T*>(m_pointer);
    }

    T* ptr(OwnerType& owner) RETURNS_NONNULL { &get(owner); }
    T* ptr(const OwnerType& owner) const RETURNS_NONNULL { return &get(owner); }

    template<typename Func>
    void initLater(const Func&)
    {
        static_assert(alignof(T) >= 4);
        static_assert(isStatelessLambda<Func>());
        // Logically we just want to stuff the function pointer into m_pointer, but then we'd be sad
        // because a function pointer is not guaranteed to be a multiple of anything. The tag bits
        // may be used for things. We address this problem by indirecting through a global const
        // variable. The "theFunc" variable is guaranteed to be native-aligned, i.e. at least a
        // multiple of 4.
        static constexpr FuncType theFunc = &callFunc<Func>;
        m_pointer = lazyTag | std::bit_cast<uintptr_t>(&theFunc);
    }

    void set(Ref<T>&& ref)
    {
        Ref<T> local = WTFMove(ref);
        m_pointer = std::bit_cast<uintptr_t>(&local.leakRef());
    }

private:
    static const uintptr_t lazyTag = 1;
    static const uintptr_t initializingTag = 2;

    template<typename Func>
    static T* callFunc(OwnerType& owner, LazyRef& ref)
    {
        ref.m_pointer |= initializingTag;
        callStatelessLambda<void, Func>(owner, ref);
        RELEASE_ASSERT(!(ref.m_pointer & lazyTag));
        RELEASE_ASSERT(!(ref.m_pointer & initializingTag));
        return std::bit_cast<T*>(ref.m_pointer);
    }

    uintptr_t m_pointer { 0 };
};

} // namespace WTF

using WTF::LazyRef;
