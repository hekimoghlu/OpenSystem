/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

#include <tuple>
#include <type_traits>
#include <variant>
#include <wtf/Assertions.h>
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/TypeTraits.h>
#include <wtf/text/WTFString.h>

namespace WTF {

struct CrossThreadCopierBaseHelper {
    template<typename T> struct RemovePointer {
        typedef T Type;
    };
    template<typename T> struct RemovePointer<T*> {
        typedef T Type;
    };

    template<typename T> struct RemovePointer<RefPtr<T>> {
        typedef T Type;
    };

    template<typename T> struct RemovePointer<Ref<T>> {
        typedef T Type;
    };

    template<typename T> struct IsEnumOrConvertibleToInteger {
        static constexpr bool value = std::is_integral<T>::value || std::is_enum<T>::value || std::is_convertible<T, long double>::value;
    };

    template<typename T> struct IsThreadSafeRefCountedPointer {
        static constexpr bool value = std::is_convertible<typename RemovePointer<T>::Type*, ThreadSafeRefCountedBase*>::value;
    };
};

template<typename T> struct CrossThreadCopierPassThrough {
    static_assert(CrossThreadCopierBaseHelper::IsEnumOrConvertibleToInteger<T>::value, "CrossThreadCopierPassThrough only used for enums or integers");
    using Type = T;
    static constexpr bool IsNeeded = false;
    static Type copy(const T& parameter)
    {
        return parameter;
    }
};

template<bool isEnumOrConvertibleToInteger, bool isThreadSafeRefCounted, typename T> struct CrossThreadCopierBase;

// Integers get passed through without any changes.
template<typename T> struct CrossThreadCopierBase<true, false, T> : public CrossThreadCopierPassThrough<T> {
};

// All non-specified types passed to crossThreadCopy() must provide isolatedCopy().
template<typename T>
struct CrossThreadCopierBase<false, false, T> {
    using Type = T;
    static constexpr bool IsNeeded = true;
    static Type copy(const Type& value)
    {
        return value.isolatedCopy();
    }

    static Type copy(Type&& value)
    {
        return WTFMove(value).isolatedCopy();
    }
};

// Custom copy methods.
template<typename T, typename Deleter>
struct CrossThreadCopierBase<false, false, std::unique_ptr<T, Deleter>> {
    using Type = std::unique_ptr<T, Deleter>;
    static constexpr bool IsNeeded = false;
    static Type copy(Type&& value)
    {
        return WTFMove(value);
    }
};

template<typename T>
struct CrossThreadCopierBase<false, false, RetainPtr<T>> {
    using Type = RetainPtr<T>;
    static constexpr bool IsNeeded = false;
    static Type copy(Type&& value)
    {
        return WTFMove(value);
    }
};

template<typename T> struct CrossThreadCopierBase<false, true, T> {
    using RefCountedType = typename CrossThreadCopierBaseHelper::RemovePointer<T>::Type;
    static_assert(std::is_convertible<RefCountedType*, ThreadSafeRefCountedBase*>::value, "T is not convertible to ThreadSafeRefCounted!");

    using Type = RefPtr<RefCountedType>;
    static constexpr bool IsNeeded = false;

    static Type copy(const T& refPtr)
    {
        return refPtr;
    }
    static Type copy(T&& refPtr)
    {
        return WTFMove(refPtr);
    }
};

// Can only be moved
template<typename T> struct CrossThreadCopierBase<false, false, Ref<T>> {
    using Type = Ref<T>;
    static constexpr bool IsNeeded = false;
    static Type copy(Type&& ref)
    {
        return WTFMove(ref);
    }
};

// Can only be moved
template<typename T> struct CrossThreadCopierBase<false, false, RefPtr<T>> {
    using Type = RefPtr<T>;
    static constexpr bool IsNeeded = false;
    static Type copy(Type&& ref)
    {
        return WTFMove(ref);
    }
};

template<typename T> struct CrossThreadCopierBase<false, true, Ref<T>> {
    static_assert(std::is_convertible<T*, ThreadSafeRefCountedBase*>::value, "T is not convertible to ThreadSafeRefCounted!");
    static constexpr bool IsNeeded = false;

    using Type = Ref<T>;
    static Type copy(const Type& ref)
    {
        return ref;
    }
    static Type copy(Type&& ref)
    {
        return WTFMove(ref);
    }
};

// Default specialization for AtomString of CrossThreadCopyable classes.
template<> struct CrossThreadCopierBase<false, false, AtomString> {
    using Type = String;
    static constexpr bool IsNeeded = true;
    static Type copy(const AtomString& source)
    {
        return source.string().isolatedCopy();
    }
    static Type copy(AtomString&& source)
    {
        return source.releaseString().isolatedCopy();
    }
};

template<> struct CrossThreadCopierBase<false, false, WTF::ASCIILiteral> {
    using Type = WTF::ASCIILiteral;
    static constexpr bool IsNeeded = false;
    static Type copy(const Type& source)
    {
        return source;
    }
};

template<typename T, typename U, typename V> struct CrossThreadCopierBase<false, false, ObjectIdentifierGeneric<T, U, V>> {
    using Type = ObjectIdentifierGeneric<T, U, V>;
    static constexpr bool IsNeeded = false;
    static Type copy(const Type& source)
    {
        return source;
    }
};

template<typename T>
struct CrossThreadCopier : public CrossThreadCopierBase<CrossThreadCopierBaseHelper::IsEnumOrConvertibleToInteger<T>::value, CrossThreadCopierBaseHelper::IsThreadSafeRefCountedPointer<T>::value, T> {
};

// Default specialization for Vectors of CrossThreadCopyable classes.
template<typename T, size_t inlineCapacity, typename OverflowHandler, size_t minCapacity> struct CrossThreadCopierBase<false, false, Vector<T, inlineCapacity, OverflowHandler, minCapacity>> {
    using Type = Vector<T, inlineCapacity, OverflowHandler, minCapacity>;
    static constexpr bool IsNeeded = CrossThreadCopier<T>::IsNeeded;
    static Type copy(const Type& source)
    {
        return WTF::map<inlineCapacity, OverflowHandler, minCapacity>(source, [](auto& object) {
            return CrossThreadCopier<T>::copy(object);
        });
    }
    static Type copy(Type&& source)
    {
        for (auto& item : std::forward<Type>(source))
            item = CrossThreadCopier<T>::copy(WTFMove(item));
        return WTFMove(source);
    }
};

// Default specialization for HashSets of CrossThreadCopyable classes
template<typename T, typename HashFunctions, typename Traits, typename TableTraits, ShouldValidateKey shouldValidateKey> struct CrossThreadCopierBase<false, false, HashSet<T, HashFunctions, Traits, TableTraits, shouldValidateKey>> {
    using Type = HashSet<T, HashFunctions, Traits, TableTraits, shouldValidateKey>;
    static constexpr bool IsNeeded = CrossThreadCopier<T>::IsNeeded;
    static Type copy(const Type& source)
    {
        Type destination;
        for (auto& object : source)
            destination.add(CrossThreadCopier<T>::copy(object));
        return destination;
    }
    static Type copy(Type&& source)
    {
        Type destination;
        destination.reserveInitialCapacity(source.size());
        while (!source.isEmpty())
            destination.add(CrossThreadCopier<T>::copy(source.takeAny()));
        return destination;
    }
};

template<typename T, typename U, typename V>
struct CrossThreadCopierBase<false, false, HashSet<ObjectIdentifierGeneric<T, U, V>>> {
    typedef HashSet<ObjectIdentifierGeneric<T, U, V>> Type;
    static Type copy(const Type& identifiers) { return identifiers; }
};

// Default specialization for HashMaps of CrossThreadCopyable classes
template<typename KeyArg, typename MappedArg, typename HashArg, typename KeyTraitsArg, typename MappedTraitsArg, typename TableTraitsArg, ShouldValidateKey shouldValidateKey>
struct CrossThreadCopierBase<false, false, HashMap<KeyArg, MappedArg, HashArg, KeyTraitsArg, MappedTraitsArg, TableTraitsArg, shouldValidateKey>> {
    using Type = HashMap<KeyArg, MappedArg, HashArg, KeyTraitsArg, MappedTraitsArg, TableTraitsArg, shouldValidateKey>;
    static constexpr bool IsNeeded = CrossThreadCopier<KeyArg>::IsNeeded || CrossThreadCopier<MappedArg>::IsNeeded;
    static Type copy(const Type& source)
    {
        Type destination;
        for (auto& [key, value] : source)
            destination.add(CrossThreadCopier<KeyArg>::copy(key), CrossThreadCopier<MappedArg>::copy(value));
        return destination;
    }
    static Type copy(Type&& source)
    {
        for (auto iterator = source.begin(), end = source.end(); iterator != end; ++iterator) {
            iterator->key = CrossThreadCopier<KeyArg>::copy(WTFMove(iterator->key));
            iterator->value = CrossThreadCopier<MappedArg>::copy(WTFMove(iterator->value));
        }
        return WTFMove(source);
    }
};

// Default specialization for pairs of CrossThreadCopyable classes
template<typename F, typename S> struct CrossThreadCopierBase<false, false, std::pair<F, S> > {
    using Type = std::pair<F, S>;
    static constexpr bool IsNeeded = CrossThreadCopier<F>::IsNeeded || CrossThreadCopier<S>::IsNeeded;
    template<typename U> static Type copy(U&& source)
    {
        return std::make_pair(CrossThreadCopier<F>::copy(std::get<0>(std::forward<U>(source))), CrossThreadCopier<S>::copy(std::get<1>(std::forward<U>(source))));
    }
};

// Default specialization for std::optional of CrossThreadCopyable class.
template<typename T> struct CrossThreadCopierBase<false, false, std::optional<T>> {
    using Type = std::optional<T>;
    static constexpr bool IsNeeded = CrossThreadCopier<T>::IsNeeded;
    template<typename U> static std::optional<T> copy(U&& source)
    {
        if (!source)
            return std::nullopt;
        return CrossThreadCopier<T>::copy(std::forward<U>(source).value());
    }
};

// Default specialization for Markable of CrossThreadCopyable class.
template<typename T, typename U> struct CrossThreadCopierBase<false, false, Markable<T, U>> {
    using Type = Markable<T, U>;
    static constexpr bool IsNeeded = CrossThreadCopier<T>::IsNeeded;
    template<typename V> static Type copy(V&& source)
    {
        if (!source)
            return std::nullopt;
        return CrossThreadCopier<T>::copy(std::forward<V>(source).value());
    }
};

template<> struct CrossThreadCopierBase<false, false, std::nullptr_t> {
    static constexpr bool IsNeeded = false;
    static std::nullptr_t copy(std::nullptr_t) { return nullptr; }
};

// Default specialization for std::variant of CrossThreadCopyable classes.
template<typename... Types> struct CrossThreadCopierBase<false, false, std::variant<Types...>> {
    using Type = std::variant<Types...>;
    static constexpr bool IsNeeded = (CrossThreadCopier<std::remove_cvref_t<Types>>::IsNeeded || ...);
    static std::variant<Types...> copy(const Type& source)
    {
        return std::visit([] (auto& type) -> std::variant<Types...> {
            return CrossThreadCopier<std::remove_cvref_t<decltype(type)>>::copy(type);
        }, source);
    }
    static std::variant<Types...> copy(Type&& source)
    {
        return std::visit([] (auto&& type) -> std::variant<Types...> {
            return CrossThreadCopier<std::remove_cvref_t<decltype(type)>>::copy(std::forward<decltype(type)>(type));
        }, WTFMove(source));
    }
};

template<>
struct CrossThreadCopierBase<false, false, void> {
    static constexpr bool IsNeeded = false;
    using Type = void;
};

template<typename T, typename U> struct CrossThreadCopierBase<false, false, Expected<T, U> > {
    using Type = Expected<T, U>;
    static constexpr bool IsNeeded = CrossThreadCopier<std::remove_cvref_t<T>>::IsNeeded || CrossThreadCopier<std::remove_cvref_t<U>>::IsNeeded;
    static Type copy(const Type& source)
    {
        if (source.has_value()) {
            if constexpr (std::is_void_v<T>)
                return source;
            else
                return CrossThreadCopier<T>::copy(source.value());
        }
        return Unexpected<U>(CrossThreadCopier<U>::copy(source.error()));
    }

    static Type copy(Type&& source)
    {
        if (source.has_value()) {
            if constexpr (std::is_void_v<T>)
                return WTFMove(source);
            else
                return CrossThreadCopier<T>::copy(WTFMove(source.value()));
        }
        return Unexpected<U>(CrossThreadCopier<U>::copy(WTFMove(source.error())));
    }
};

// Default specialization for std::tuple of CrossThreadCopyable classes.
template<typename... Types> struct CrossThreadCopierBase<false, false, std::tuple<Types...>> {
    using Type = std::tuple<Types...>;
    static constexpr bool IsNeeded = (CrossThreadCopier<std::remove_cvref_t<Types>>::IsNeeded || ...);
    static Type copy(const Type& source)
    {
        return std::apply([]<typename ...Ts>(Ts const&... ts) {
            return std::make_tuple((CrossThreadCopier<std::remove_cvref_t<Ts>>::copy(ts), ...));
        }, source);
    }
    static Type copy(Type&& source)
    {
        return std::apply([]<typename ...Ts>(Ts&&... ts) {
            return std::make_tuple((CrossThreadCopier<std::remove_cvref_t<Ts>>::copy(WTFMove(ts)), ...));
        }, WTFMove(source));
    }
};

template<typename T> auto crossThreadCopy(T&& source)
{
    return CrossThreadCopier<std::remove_cvref_t<T>>::copy(std::forward<T>(source));
}

} // namespace WTF

using WTF::CrossThreadCopierBaseHelper;
using WTF::CrossThreadCopierBase;
using WTF::CrossThreadCopier;
using WTF::crossThreadCopy;
