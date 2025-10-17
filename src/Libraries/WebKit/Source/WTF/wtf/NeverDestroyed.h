/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

#include <type_traits>
#include <utility>
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/MainThread.h>
#include <wtf/RefCounted.h>

// NeverDestroyed is a smart-pointer-like class that ensures that the destructor
// for the given object is never called, but doesn't use the heap to allocate it.
// It's useful for static local variables, and can be used like so:
//
// MySharedGlobal& mySharedGlobal()
// {
//   static NeverDestroyed<MySharedGlobal> myGlobal("Hello", 42);
//   return myGlobal;
// }

namespace WTF {

struct AnyThreadsAccessTraits {
    static void assertAccess()
    {
    }
};

struct MainThreadAccessTraits {
    static void assertAccess()
    {
        ASSERT(isMainThread());
    }
};

template<typename T, typename AccessTraits> class NeverDestroyed {
    WTF_MAKE_NONCOPYABLE(NeverDestroyed);
    WTF_FORBID_HEAP_ALLOCATION;
public:

    template<typename... Args> NeverDestroyed(Args&&... args)
    {
        AccessTraits::assertAccess();
        MaybeRelax<T>(new (storagePointer()) T(std::forward<Args>(args)...));
    }

    NeverDestroyed(NeverDestroyed&& other)
    {
        AccessTraits::assertAccess();
        MaybeRelax<T>(new (storagePointer()) T(WTFMove(*other.storagePointer())));
    }

    operator T&() { return *storagePointer(); }
    T& get() { return *storagePointer(); }

    T* operator->() { return storagePointer(); }

    operator const T&() const { return *storagePointer(); }
    const T& get() const { return *storagePointer(); }

    const T* operator->() const { return storagePointer(); }

private:
    using PointerType = typename std::remove_const<T>::type*;

    PointerType storagePointer() const
    {
        AccessTraits::assertAccess();
        return const_cast<PointerType>(reinterpret_cast<const T*>(&m_storage));
    }

    template<typename PtrType, bool ShouldRelax = std::is_base_of<RefCountedBase, PtrType>::value> struct MaybeRelax {
        explicit MaybeRelax(PtrType*) { }
    };
    template<typename PtrType> struct MaybeRelax<PtrType, true> {
        explicit MaybeRelax(PtrType* ptr) { ptr->relaxAdoptionRequirement(); }
    };

    // FIXME: Investigate whether we should allocate a hunk of virtual memory
    // and hand out chunks of it to NeverDestroyed instead, to reduce fragmentation.
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type m_storage;
    ALLOW_DEPRECATED_DECLARATIONS_END
};

// FIXME: It's messy to have to repeat the whole class just to make this "lazy" version.
// Should revisit clients to see if we really need this, and perhaps use templates to
// share more of the code with the main NeverDestroyed above.
template<typename T, typename AccessTraits> class LazyNeverDestroyed {
    WTF_MAKE_NONCOPYABLE(LazyNeverDestroyed);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    LazyNeverDestroyed() = default;

    template<typename... Args>
    void construct(Args&&... args)
    {
        AccessTraits::assertAccess();
        constructWithoutAccessCheck(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void constructWithoutAccessCheck(Args&&... args)
    {
        ASSERT(!m_isConstructed);
#if ASSERT_ENABLED
        m_isConstructed = true;
#endif
        MaybeRelax<T>(new (storagePointerWithoutAccessCheck()) T(std::forward<Args>(args)...));
    }

    operator T&() { return *storagePointer(); }
    T& get() { return *storagePointer(); }

    T* operator->() { return storagePointer(); }

    operator const T&() const { return *storagePointer(); }
    const T& get() const { return *storagePointer(); }

    const T* operator->() const { return storagePointer(); }

#if ASSERT_ENABLED
    bool isConstructed() const { return m_isConstructed; }
#endif

private:
    using PointerType = typename std::remove_const<T>::type*;

    PointerType storagePointerWithoutAccessCheck() const
    {
        ASSERT(m_isConstructed);
        return const_cast<PointerType>(reinterpret_cast<const T*>(&m_storage));
    }

    PointerType storagePointer() const
    {
        AccessTraits::assertAccess();
        return storagePointerWithoutAccessCheck();
    }

    template<typename PtrType, bool ShouldRelax = std::is_base_of<RefCountedBase, PtrType>::value> struct MaybeRelax {
        explicit MaybeRelax(PtrType*) { }
    };
    template<typename PtrType> struct MaybeRelax<PtrType, true> {
        explicit MaybeRelax(PtrType* ptr) { ptr->relaxAdoptionRequirement(); }
    };

#if ASSERT_ENABLED
    // LazyNeverDestroyed objects are always static, so this variable is initialized to false.
    // It must not be initialized dynamically; that would not be thread safe.
    bool m_isConstructed;
#endif

    // FIXME: Investigate whether we should allocate a hunk of virtual memory
    // and hand out chunks of it to NeverDestroyed instead, to reduce fragmentation.
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type m_storage;
    ALLOW_DEPRECATED_DECLARATIONS_END
};

template<typename T> NeverDestroyed(T) -> NeverDestroyed<T>;

template<typename T> using MainThreadNeverDestroyed = NeverDestroyed<T, MainThreadAccessTraits>;

template<typename T> using MainThreadLazyNeverDestroyed = LazyNeverDestroyed<T, MainThreadAccessTraits>;

} // namespace WTF;

using WTF::LazyNeverDestroyed;
using WTF::NeverDestroyed;
using WTF::MainThreadNeverDestroyed;
using WTF::MainThreadLazyNeverDestroyed;
using WTF::AnyThreadsAccessTraits;
using WTF::MainThreadAccessTraits;
