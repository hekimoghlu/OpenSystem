/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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

#include <wtf/CompactRefPtrTuple.h>
#include <wtf/Packed.h>
#include <wtf/WeakRef.h>

namespace WTF {

#define USING_CAN_MAKE_WEAKPTR(BASE) \
    using BASE::weakImpl; \
    using BASE::weakImplIfExists; \
    using BASE::weakCount; \
    using BASE::WeakValueType; \
    using BASE::WeakPtrImplType;

// Note: you probably want to inherit from CanMakeWeakPtr rather than use this directly.
template<typename T, typename WeakPtrImpl = DefaultWeakPtrImpl>
class WeakPtrFactory {
    WTF_MAKE_NONCOPYABLE(WeakPtrFactory);
    WTF_MAKE_FAST_ALLOCATED;
public:
    using ObjectType = T;
    using WeakPtrImplType = WeakPtrImpl;

    WeakPtrFactory()
#if ASSERT_ENABLED
        : m_wasConstructedOnMainThread(isMainThread())
#endif
    {
    }

    void prepareForUseOnlyOnNonMainThread()
    {
#if ASSERT_ENABLED
        ASSERT(m_wasConstructedOnMainThread);
        m_wasConstructedOnMainThread = false;
#endif
    }

    ~WeakPtrFactory()
    {
        if (m_impl)
            m_impl->clear();
    }

    WeakPtrImpl* impl() const
    {
        return m_impl.get();
    }

    void initializeIfNeeded(const T& object) const
    {
        if (m_impl)
            return;

        ASSERT(m_wasConstructedOnMainThread == isMainThread());

        static_assert(std::is_final_v<WeakPtrImpl>);
        m_impl = adoptRef(*new WeakPtrImpl(const_cast<T*>(&object)));
    }

    template<typename U> WeakPtr<U, WeakPtrImpl> createWeakPtr(U& object, EnableWeakPtrThreadingAssertions enableWeakPtrThreadingAssertions = EnableWeakPtrThreadingAssertions::Yes) const
    {
        initializeIfNeeded(object);

        ASSERT(&object == m_impl->template get<T>());
        return WeakPtr<U, WeakPtrImpl>(*m_impl, enableWeakPtrThreadingAssertions);
    }

    void revokeAll()
    {
        if (RefPtr impl = std::exchange(m_impl, nullptr))
            impl->clear();
    }

    unsigned weakPtrCount() const
    {
        return m_impl ? m_impl->refCount() - 1 : 0u;
    }

#if ASSERT_ENABLED
    bool isInitialized() const { return m_impl; }
#endif

private:
    template<typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakHashSet;
    template<typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakListHashSet;
    template<typename, typename, typename> friend class WeakHashMap;
    template<typename, typename, typename> friend class WeakPtr;
    template<typename, typename> friend class WeakRef;

    mutable RefPtr<WeakPtrImpl> m_impl;
#if ASSERT_ENABLED
    bool m_wasConstructedOnMainThread;
#endif
};

// Note: you probably want to inherit from CanMakeWeakPtrWithBitField rather than use this directly.
template<typename T, typename WeakPtrImpl = DefaultWeakPtrImpl>
class WeakPtrFactoryWithBitField {
    WTF_MAKE_NONCOPYABLE(WeakPtrFactoryWithBitField);
    WTF_MAKE_FAST_ALLOCATED;
public:
    using ObjectType = T;
    using WeakPtrImplType = WeakPtrImpl;

    WeakPtrFactoryWithBitField()
#if ASSERT_ENABLED
        : m_wasConstructedOnMainThread(isMainThread())
#endif
    {
    }

    ~WeakPtrFactoryWithBitField()
    {
        if (auto* pointer = m_impl.pointer())
            pointer->clear();
    }

    WeakPtrImpl* impl() const
    {
        return m_impl.pointer();
    }

    void initializeIfNeeded(const T& object) const
    {
        if (m_impl.pointer())
            return;

        ASSERT(m_wasConstructedOnMainThread == isMainThread());

        static_assert(std::is_final_v<WeakPtrImpl>);
        m_impl.setPointer(adoptRef(*new WeakPtrImpl(const_cast<T*>(&object))));
    }

    template<typename U> WeakPtr<U, WeakPtrImpl> createWeakPtr(U& object, EnableWeakPtrThreadingAssertions enableWeakPtrThreadingAssertions = EnableWeakPtrThreadingAssertions::Yes) const
    {
        initializeIfNeeded(object);

        ASSERT(&object == m_impl.pointer()->template get<T>());
        return WeakPtr<U, WeakPtrImpl>(*m_impl.pointer(), enableWeakPtrThreadingAssertions);
    }

    void revokeAll()
    {
        if (auto* pointer = m_impl.pointer()) {
            pointer->clear();
            m_impl.setPointer(nullptr);
        }
    }

    unsigned weakPtrCount() const
    {
        if (auto* pointer = m_impl.pointer())
            return pointer->refCount() - 1;
        return 0;
    }

#if ASSERT_ENABLED
    bool isInitialized() const { return m_impl.pointer(); }
#endif

    uint16_t bitfield() const { return m_impl.type(); }
    void setBitfield(uint16_t value) const { return m_impl.setType(value); }

private:
    template<typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakHashSet;
    template<typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakListHashSet;
    template<typename, typename, typename> friend class WeakHashMap;
    template<typename, typename, typename> friend class WeakPtr;
    template<typename, typename> friend class WeakRef;

    mutable CompactRefPtrTuple<WeakPtrImpl, uint16_t> m_impl;
#if ASSERT_ENABLED
    bool m_wasConstructedOnMainThread;
#endif
};

// We use lazy initialization of the WeakPtrFactory by default to avoid unnecessary initialization. Eager
// initialization is however useful if you plan to call construct WeakPtrs from other threads.
enum class WeakPtrFactoryInitialization { Lazy, Eager };

} // namespace WTF

using WTF::WeakPtrFactory;
using WTF::WeakPtrFactoryInitialization;
