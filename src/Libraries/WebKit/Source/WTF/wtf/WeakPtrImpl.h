/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#include <wtf/GetPtr.h>
#include <wtf/HashTraits.h>
#include <wtf/SingleThreadIntegralWrapper.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Threading.h>
#include <wtf/TypeCasts.h>

namespace WTF {

DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(WeakPtrImplBase);

template<typename Derived>
class WeakPtrImplBase : public ThreadSafeRefCounted<Derived> {
    WTF_MAKE_NONCOPYABLE(WeakPtrImplBase);
    WTF_MAKE_FAST_COMPACT_ALLOCATED_WITH_HEAP_IDENTIFIER(WeakPtrImplBase);
public:
    ~WeakPtrImplBase() = default;

    template<typename T> typename T::WeakValueType* get()
    {
        return static_cast<typename T::WeakValueType*>(m_ptr);
    }

    explicit operator bool() const { return m_ptr; }
    void clear() { m_ptr = nullptr; }

#if ASSERT_ENABLED
    bool wasConstructedOnMainThread() const { return m_wasConstructedOnMainThread; }
#endif

    template<typename T>
    explicit WeakPtrImplBase(T* ptr)
        : m_ptr(static_cast<typename T::WeakValueType*>(ptr))
#if ASSERT_ENABLED
        , m_wasConstructedOnMainThread(isMainThread())
#endif
    {
    }

private:
    void* m_ptr;
#if ASSERT_ENABLED
    bool m_wasConstructedOnMainThread;
#endif
};

class DefaultWeakPtrImpl final : public WeakPtrImplBase<DefaultWeakPtrImpl> {
public:
    template<typename T>
    explicit DefaultWeakPtrImpl(T* ptr)
        : WeakPtrImplBase<DefaultWeakPtrImpl>(ptr)
    {
    }
};

DECLARE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(WeakPtrImplBaseSingleThread);

template<typename Derived>
class WeakPtrImplBaseSingleThread {
    WTF_MAKE_NONCOPYABLE(WeakPtrImplBaseSingleThread);
    WTF_MAKE_FAST_COMPACT_ALLOCATED_WITH_HEAP_IDENTIFIER(WeakPtrImplBaseSingleThread);
public:
    ~WeakPtrImplBaseSingleThread() = default;

    template<typename T> typename T::WeakValueType* get()
    {
        return static_cast<typename T::WeakValueType*>(m_ptr);
    }

    explicit operator bool() const { return m_ptr; }
    void clear() { m_ptr = nullptr; }

#if ASSERT_ENABLED
    bool wasConstructedOnMainThread() const { return m_wasConstructedOnMainThread; }
#endif

    template<typename T>
    explicit WeakPtrImplBaseSingleThread(T* ptr)
        : m_ptr(static_cast<typename T::WeakValueType*>(ptr))
#if ASSERT_ENABLED
        , m_wasConstructedOnMainThread(isMainThread())
#endif
    {
    }

    uint32_t refCount() const { return m_refCount; }
    void ref() const { ++m_refCount; }
    void deref() const
    {
        uint32_t tempRefCount = m_refCount - 1;
        if (!tempRefCount) {
            delete const_cast<Derived*>(static_cast<const Derived*>(this));
            return;
        }
        m_refCount = tempRefCount;
    }

private:
    mutable SingleThreadIntegralWrapper<uint32_t> m_refCount { 1 };
    void* m_ptr;
#if ASSERT_ENABLED
    bool m_wasConstructedOnMainThread;
#endif
};

class SingleThreadWeakPtrImpl final : public WeakPtrImplBaseSingleThread<SingleThreadWeakPtrImpl> {
public:
    template<typename T>
    explicit SingleThreadWeakPtrImpl(T* ptr)
        : WeakPtrImplBaseSingleThread<SingleThreadWeakPtrImpl>(ptr)
    {
    }
};

} // namespace WTF

using WTF::SingleThreadWeakPtrImpl;
