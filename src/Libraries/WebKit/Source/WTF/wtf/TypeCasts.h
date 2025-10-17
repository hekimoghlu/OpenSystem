/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

namespace WTF {

template <typename ExpectedType, typename ArgType, bool isBaseType = std::is_base_of_v<ExpectedType, ArgType>>
struct TypeCastTraits {
    static bool isOfType(ArgType&)
    {
        // If you're hitting this assertion, it is likely because you used
        // is<>() or downcast<>() with a type that doesn't have the needed
        // TypeCastTraits specialization. Please use the following macro
        // to add that specialization:
        // SPECIALIZE_TYPE_TRAITS_BEGIN() / SPECIALIZE_TYPE_TRAITS_END()
        static_assert(std::is_void_v<ExpectedType>, "Missing TypeCastTraits specialization");
        return false;
    }
};

// Template specialization for the case where ExpectedType is a base of ArgType,
// so we can return return true unconditionally.
template <typename ExpectedType, typename ArgType>
struct TypeCastTraits<ExpectedType, ArgType, true /* isBaseType */> {
    static bool isOfType(ArgType&) { return true; }
};

// Type checking function, to use before casting with downcast<>().
template <typename ExpectedType, typename ArgType>
inline bool is(const ArgType& source)
{
    static_assert(std::is_base_of_v<ArgType, ExpectedType>, "Unnecessary type check");
    return TypeCastTraits<const ExpectedType, const ArgType>::isOfType(source);
}

template <typename ExpectedType, typename ArgType>
inline bool is(ArgType* source)
{
    static_assert(std::is_base_of_v<ArgType, ExpectedType>, "Unnecessary type check");
    return source && TypeCastTraits<const ExpectedType, const ArgType>::isOfType(*source);
}

// Update T's constness to match Reference's.
template <typename Reference, typename T>
using match_constness_t =
    typename std::conditional_t<std::is_const_v<Reference>, typename std::add_const_t<T>, typename std::remove_const_t<T>>;

template<typename Target, typename Source>
inline match_constness_t<Source, Target>& uncheckedDowncast(Source& source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    ASSERT_WITH_SECURITY_IMPLICATION(is<Target>(source));
    SUPPRESS_MEMORY_UNSAFE_CAST return static_cast<match_constness_t<Source, Target>&>(source);
}

template<typename Target, typename Source>
inline match_constness_t<Source, Target>* uncheckedDowncast(Source* source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    ASSERT_WITH_SECURITY_IMPLICATION(!source || is<Target>(*source));
    SUPPRESS_MEMORY_UNSAFE_CAST return static_cast<match_constness_t<Source, Target>*>(source);
}

template<typename Target, typename Source>
inline match_constness_t<Source, Target>& downcast(Source& source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    RELEASE_ASSERT(is<Target>(source));
    SUPPRESS_MEMORY_UNSAFE_CAST return static_cast<match_constness_t<Source, Target>&>(source);
}

template<typename Target, typename Source>
inline match_constness_t<Source, Target>* downcast(Source* source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    RELEASE_ASSERT(!source || is<Target>(*source));
    SUPPRESS_MEMORY_UNSAFE_CAST return static_cast<match_constness_t<Source, Target>*>(source);
}

template<typename Target, typename Source>
inline match_constness_t<Source, Target>* dynamicDowncast(Source& source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    SUPPRESS_MEMORY_UNSAFE_CAST return is<Target>(source) ? &static_cast<match_constness_t<Source, Target>&>(source) : nullptr;
}

template<typename Target, typename Source>
inline match_constness_t<Source, Target>* dynamicDowncast(Source* source)
{
    static_assert(!std::is_same_v<Source, Target>, "Unnecessary cast to same type");
    static_assert(std::is_base_of_v<Source, Target>, "Should be a downcast");
    SUPPRESS_MEMORY_UNSAFE_CAST return is<Target>(source) ? static_cast<match_constness_t<Source, Target>*>(source) : nullptr;
}

// Add support for type checking / casting using is<>() / downcast<>() helpers for a specific class.
#define SPECIALIZE_TYPE_TRAITS_BEGIN(ClassName) \
namespace WTF { \
template <typename ArgType> \
class TypeCastTraits<const ClassName, ArgType, false /* isBaseType */> { \
public: \
    static bool isOfType(ArgType& source) { return isType(source); } \
private:

#define SPECIALIZE_TYPE_TRAITS_END() \
}; \
}

// Explicit specialization for C++ standard library types.

template<typename ExpectedType, typename ArgType, typename Deleter>
inline bool is(std::unique_ptr<ArgType, Deleter>& source)
{
    return is<ExpectedType>(source.get());
}

template<typename ExpectedType, typename ArgType, typename Deleter>
inline bool is(const std::unique_ptr<ArgType, Deleter>& source)
{
    return is<ExpectedType>(source.get());
}

} // namespace WTF

using WTF::TypeCastTraits;
using WTF::is;
using WTF::downcast;
using WTF::dynamicDowncast;
using WTF::uncheckedDowncast;
