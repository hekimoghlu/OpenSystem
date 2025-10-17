/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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

#include <concepts>

namespace WebCore {

// Types can specialize this and set the value to true to be treated as "empty-like"
// for CSS value type algorithms.
// Requirements: None.
template<typename> inline constexpr auto TreatAsEmptyLike = false;

// The `EmptyLike` concept can be used to filter to types that specialize `TreatAsEmptyLike`.
template<typename T> concept EmptyLike = TreatAsEmptyLike<T>;

// Types can specialize this and set the value to true to be treated as "optional-like"
// for CSS value type algorithms.
// Requirements: Types be comparable to bool and have a operator* function.
template<typename> inline constexpr auto TreatAsOptionalLike = false;

// The `OptionalLike` concept can be used to filter to types that specialize `TreatAsOptionalLike`.
template<typename T> concept OptionalLike = TreatAsOptionalLike<T>;

// Types can specialize this and set the value to true to be treated as "tuple-like"
// for CSS value type algorithms.
// NOTE: This gets automatically specialized when using the *_TUPLE_LIKE_CONFORMANCE macros.
// Requirements: Types must have conform the to the standard tuple-like pseudo-protocol.
template<typename> inline constexpr auto TreatAsTupleLike = false;

// The `TupleLike` concept can be used to filter to types that specialize `TreatAsTupleLike`.
template<typename T> concept TupleLike = TreatAsTupleLike<T>;

// Types can specialize this and set the value to true to be treated as "range-like"
// for CSS value type algorithms.
// Requirements: Types must have valid begin()/end() functions.
template<typename> inline constexpr auto TreatAsRangeLike = false;

// The `RangeLike` concept can be used to filter to types that specialize `TreatAsRangeLike`.
template<typename T> concept RangeLike = TreatAsRangeLike<T>;

// Types can specialize this and set the value to true to be treated as "variant-like"
// for CSS value type algorithms.
// Requirements: Types must be able to be passed to WTF::switchOn().
template<typename> inline constexpr auto TreatAsVariantLike = false;

// The `VariantLike` concept can be used to filter to types that specialize `TreatAsVariantLike`.
template<typename T> concept VariantLike = TreatAsVariantLike<T>;

// The `HasIsZero` concept can be used to filter to types that have an `isZero` member function.
template<typename T> concept HasIsZero = requires(T t) {
    { t.isZero() } -> std::convertible_to<bool>;
};

// The `HasIsEmpty` concept can be used to filter to types that have an `isEmpty` member function.
template<typename T> concept HasIsEmpty = requires(T t) {
    { t.isEmpty() } -> std::convertible_to<bool>;
};

} // namespace WebCore
