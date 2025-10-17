/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include <wtf/TypeTraits.h>
#include <wtf/WeakPtrImpl.h>

namespace WTF {
// Classes that offer weak pointers should also offer RefPtr or CheckedPtr. Please do not add new exceptions.
template<typename T> struct IsDeprecatedWeakRefSmartPointerException : std::false_type { };

enum class EnableWeakPtrThreadingAssertions : bool { No, Yes };

// Similar to a WeakPtr but it is an error for it to become null. It is useful for hardening when replacing
// things like `Foo& m_foo`. It is similar to CheckedRef but it generates crashes that are more actionable.
template<typename T, typename WeakPtrImpl>
class WeakRef {
public:
    template<typename = std::enable_if_t<!IsSmartPtr<T>::value && !std::is_pointer_v<T>>>
    WeakRef(const T& object, EnableWeakPtrThreadingAssertions shouldEnableAssertions = EnableWeakPtrThreadingAssertions::Yes)
        : m_impl(object.weakImpl())
#if ASSERT_ENABLED
        , m_shouldEnableAssertions(shouldEnableAssertions == EnableWeakPtrThreadingAssertions::Yes)
#endif
    {
        UNUSED_PARAM(shouldEnableAssertions);
    }

    explicit WeakRef(Ref<WeakPtrImpl>&& impl, EnableWeakPtrThreadingAssertions shouldEnableAssertions = EnableWeakPtrThreadingAssertions::Yes)
        : m_impl(WTFMove(impl))
#if ASSERT_ENABLED
        , m_shouldEnableAssertions(shouldEnableAssertions == EnableWeakPtrThreadingAssertions::Yes)
#endif
    {
        UNUSED_PARAM(shouldEnableAssertions);
    }

    WeakRef(HashTableDeletedValueType) : m_impl(HashTableDeletedValue) { }
    WeakRef(HashTableEmptyValueType) : m_impl(HashTableEmptyValue) { }

    bool isHashTableDeletedValue() const { return m_impl.isHashTableDeletedValue(); }
    bool isHashTableEmptyValue() const { return m_impl.isHashTableEmptyValue(); }

    WeakPtrImpl& impl() const { return m_impl; }
    Ref<WeakPtrImpl> releaseImpl() { return WTFMove(m_impl); }

    T* ptrAllowingHashTableEmptyValue() const
    {
        static_assert(
            HasRefPtrMemberFunctions<T>::value || HasCheckedPtrMemberFunctions<T>::value || IsDeprecatedWeakRefSmartPointerException<std::remove_cv_t<T>>::value,
            "Classes that offer weak pointers should also offer RefPtr or CheckedPtr. Please do not add new exceptions.");

        return !m_impl.isHashTableEmptyValue() ? static_cast<T*>(m_impl->template get<T>()) : nullptr;
    }

    T* ptr() const
    {
        static_assert(
            HasRefPtrMemberFunctions<T>::value || HasCheckedPtrMemberFunctions<T>::value || IsDeprecatedWeakRefSmartPointerException<std::remove_cv_t<T>>::value,
            "Classes that offer weak pointers should also offer RefPtr or CheckedPtr. Please do not add new exceptions.");

        auto* ptr = static_cast<T*>(m_impl->template get<T>());
        RELEASE_ASSERT(ptr);
        return ptr;
    }

    T& get() const
    {
        static_assert(
            HasRefPtrMemberFunctions<T>::value || HasCheckedPtrMemberFunctions<T>::value || IsDeprecatedWeakRefSmartPointerException<std::remove_cv_t<T>>::value,
            "Classes that offer weak pointers should also offer RefPtr or CheckedPtr. Please do not add new exceptions.");

        auto* ptr = static_cast<T*>(m_impl->template get<T>());
        RELEASE_ASSERT(ptr);
        return *ptr;
    }

    operator T&() const { return get(); }

    T* operator->() const
    {
        ASSERT(canSafelyBeUsed());
        return ptr();
    }

    EnableWeakPtrThreadingAssertions enableWeakPtrThreadingAssertions() const
    {
#if ASSERT_ENABLED
        return m_shouldEnableAssertions ? EnableWeakPtrThreadingAssertions::Yes : EnableWeakPtrThreadingAssertions::No;
#else
        return EnableWeakPtrThreadingAssertions::No;
#endif
    }

private:
#if ASSERT_ENABLED
    inline bool canSafelyBeUsed() const
    {
        // FIXME: Our GC threads currently need to get opaque pointers from WeakPtrs and have to be special-cased.
        return !m_impl
            || !m_shouldEnableAssertions
            || (m_impl->wasConstructedOnMainThread() && Thread::mayBeGCThread())
            || m_impl->wasConstructedOnMainThread() == isMainThread();
    }
#endif

    Ref<WeakPtrImpl> m_impl;
#if ASSERT_ENABLED
    bool m_shouldEnableAssertions { true };
#endif
};

template<class T, typename = std::enable_if_t<!IsSmartPtr<T>::value && !std::is_pointer_v<T>>>
WeakRef(const T& value, EnableWeakPtrThreadingAssertions = EnableWeakPtrThreadingAssertions::Yes) -> WeakRef<T, typename T::WeakPtrImplType>;

template <typename T, typename WeakPtrImpl>
struct GetPtrHelper<WeakRef<T, WeakPtrImpl>> {
    using PtrType = T*;
    using UnderlyingType = T;
    static T* getPtr(const WeakRef<T, WeakPtrImpl>& p) { return const_cast<T*>(p.ptr()); }
};

template <typename T, typename WeakPtrImpl>
struct IsSmartPtr<WeakRef<T, WeakPtrImpl>> {
    static constexpr bool value = true;
    static constexpr bool isNullable = false;
};

template<typename P, typename WeakPtrImpl> struct WeakRefHashTraits : SimpleClassHashTraits<WeakRef<P, WeakPtrImpl>> {
    static constexpr bool emptyValueIsZero = true;
    static WeakRef<P, WeakPtrImpl> emptyValue() { return HashTableEmptyValue; }

    template <typename>
    static void constructEmptyValue(WeakRef<P, WeakPtrImpl>& slot)
    {
        new (NotNull, std::addressof(slot)) WeakRef<P, WeakPtrImpl>(HashTableEmptyValue);
    }

    static constexpr bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const WeakRef<P, WeakPtrImpl>& value) { return value.isHashTableEmptyValue(); }

    using PeekType = P*;
    static PeekType peek(const WeakRef<P, WeakPtrImpl>& value) { return const_cast<PeekType>(value.ptrAllowingHashTableEmptyValue()); }
    static PeekType peek(P* value) { return value; }

    using TakeType = WeakPtr<P, WeakPtrImpl>;
    static TakeType take(WeakRef<P, WeakPtrImpl>&& value) { return isEmptyValue(value) ? nullptr : WeakPtr<P, WeakPtrImpl>(WTFMove(value)); }
};

template<typename P, typename WeakPtrImpl> struct HashTraits<WeakRef<P, WeakPtrImpl>> : WeakRefHashTraits<P, WeakPtrImpl> { };

template<typename P, typename WeakPtrImpl> struct PtrHash<WeakRef<P, WeakPtrImpl>> : PtrHashBase<WeakRef<P, WeakPtrImpl>, IsSmartPtr<WeakRef<P, WeakPtrImpl>>::value> {
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<typename P, typename WeakPtrImpl> struct DefaultHash<WeakRef<P, WeakPtrImpl>> : PtrHash<WeakRef<P, WeakPtrImpl>> { };

template<typename T> using SingleThreadWeakRef = WeakRef<T, SingleThreadWeakPtrImpl>;

template<typename ExpectedType, typename ArgType, typename WeakPtrImpl>
inline bool is(WeakRef<ArgType, WeakPtrImpl>& source)
{
    return is<ExpectedType>(source.get());
}

template<typename ExpectedType, typename ArgType, typename WeakPtrImpl>
inline bool is(const WeakRef<ArgType, WeakPtrImpl>& source)
{
    return is<ExpectedType>(source.get());
}

template<typename Target, typename Source, typename WeakPtrImpl>
inline WeakRef<match_constness_t<Source, Target>, WeakPtrImpl> downcast(WeakRef<Source, WeakPtrImpl> source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    RELEASE_ASSERT(is<Target>(source));
    return WeakRef<match_constness_t<Source, Target>, WeakPtrImpl> { static_reference_cast<match_constness_t<Source, Target>>(source.releaseImpl()), source.enableWeakPtrThreadingAssertions() };
}

template<typename Target, typename Source, typename WeakPtrImpl>
inline WeakPtr<match_constness_t<Source, Target>, WeakPtrImpl> dynamicDowncast(WeakRef<Source, WeakPtrImpl> source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    if (!is<Target>(source))
        return nullptr;
    return WeakPtr<match_constness_t<Source, Target>, WeakPtrImpl> { static_reference_cast<match_constness_t<Source, Target>>(source.releaseImpl()), source.enableWeakPtrThreadingAssertions() };
}

} // namespace WTF

using WTF::EnableWeakPtrThreadingAssertions;
using WTF::SingleThreadWeakRef;
using WTF::WeakRef;
