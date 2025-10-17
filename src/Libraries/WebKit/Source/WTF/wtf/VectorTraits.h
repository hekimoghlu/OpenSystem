/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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

#include <memory>
#include <type_traits>
#include <utility>
#include <wtf/RefPtr.h>
#include <wtf/UniqueRef.h>

namespace WTF {

    class AtomString;

    template<bool isPod, typename T>
    struct VectorTraitsBase;

    template<typename T>
    struct VectorTraitsBase<false, T>
    {
        static constexpr bool needsInitialization = true;
        static constexpr bool canInitializeWithMemset = false;
        static constexpr bool canMoveWithMemcpy = false;
        static constexpr bool canCopyWithMemcpy = false;
        static constexpr bool canFillWithMemset = false;
        static constexpr bool canCompareWithMemcmp = false;
    };

    template<typename T>
    struct VectorTraitsBase<true, T>
    {
        static constexpr bool needsInitialization = false;
        static constexpr bool canInitializeWithMemset = true;
        static constexpr bool canMoveWithMemcpy = true;
        static constexpr bool canCopyWithMemcpy = true;
        static constexpr bool canFillWithMemset = sizeof(T) == sizeof(char) && std::is_integral<T>::value;
        static constexpr bool canCompareWithMemcmp = true;
    };

    template<typename T>
    struct VectorTraits : VectorTraitsBase<std::is_standard_layout_v<T> && std::is_trivial_v<T>, T> { };

    struct SimpleClassVectorTraits : VectorTraitsBase<false, void>
    {
        static constexpr bool canInitializeWithMemset = true;
        static constexpr bool canMoveWithMemcpy = true;
        static constexpr bool canCompareWithMemcmp = true;
    };

    // We know smart pointers are simple enough that initializing to 0 and moving with memcpy
    // (and then not destructing the original) will work.

    template<typename P> struct VectorTraits<RefPtr<P>> : SimpleClassVectorTraits { };
    template<typename P> struct VectorTraits<std::unique_ptr<P>> : SimpleClassVectorTraits { };
    template<typename P> struct VectorTraits<UniqueRef<P>> : SimpleClassVectorTraits { };
    template<typename P> struct VectorTraits<std::reference_wrapper<P>> : SimpleClassVectorTraits { };
    template<typename P> struct VectorTraits<Ref<P>> : SimpleClassVectorTraits { };
    template<> struct VectorTraits<AtomString> : SimpleClassVectorTraits { };

    template<> struct VectorTraits<ASCIILiteral> : VectorTraitsBase<false, void> {
        static constexpr bool canInitializeWithMemset = true;
        static constexpr bool canMoveWithMemcpy = true;
        static constexpr bool canCompareWithMemcmp = false;
    };

    template<typename First, typename Second>
    struct VectorTraits<std::pair<First, Second>>
    {
        typedef VectorTraits<First> FirstTraits;
        typedef VectorTraits<Second> SecondTraits;

        static constexpr bool needsInitialization = FirstTraits::needsInitialization || SecondTraits::needsInitialization;
        static constexpr bool canInitializeWithMemset = FirstTraits::canInitializeWithMemset && SecondTraits::canInitializeWithMemset;
        static constexpr bool canMoveWithMemcpy = FirstTraits::canMoveWithMemcpy && SecondTraits::canMoveWithMemcpy;
        static constexpr bool canCopyWithMemcpy = FirstTraits::canCopyWithMemcpy && SecondTraits::canCopyWithMemcpy;
        static constexpr bool canFillWithMemset = false;
        static constexpr bool canCompareWithMemcmp = FirstTraits::canCompareWithMemcmp && SecondTraits::canCompareWithMemcmp;
    };

} // namespace WTF

using WTF::VectorTraits;
using WTF::SimpleClassVectorTraits;
