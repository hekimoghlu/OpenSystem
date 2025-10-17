/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// <steps-easing-function> = steps( <integer>, <steps-easing-function-position>? )
// <steps-easing-function-position> = jump-start | jump-end | jump-none | jump-both | start | end
//
// with range constraints, this is:
//
// <steps-easing-function> = steps( <integer [1,âˆž]>, jump-start )
//                         | steps( <integer [1,âˆž]>, jump-end )
//                         | steps( <integer [1,âˆž]>, jump-both )
//                         | steps( <integer [1,âˆž]>, start )
//                         | steps( <integer [1,âˆž]>, end )
//                         | steps( <integer [2,âˆž]>, jump-none )
// https://drafts.csswg.org/css-easing-2/#funcdef-steps
struct StepsEasingParameters {
    enum class ShouldSerializeKeyword : bool { No, Yes };

    template<typename T, typename Keyword, auto shouldSerializeKeyword = ShouldSerializeKeyword::Yes>
    struct Kind {
        static constexpr Keyword keyword = Keyword { };
        T steps;

        bool operator==(const Kind&) const = default;
    };

    using JumpStart = Kind<Integer<Range{1,Range::infinity}>, Keyword::JumpStart>;
    using JumpEnd   = Kind<Integer<Range{1,Range::infinity}>, Keyword::JumpEnd, ShouldSerializeKeyword::No>;
    using JumpBoth  = Kind<Integer<Range{1,Range::infinity}>, Keyword::JumpBoth>;
    using Start     = Kind<Integer<Range{1,Range::infinity}>, Keyword::Start>;
    using End       = Kind<Integer<Range{1,Range::infinity}>, Keyword::End, ShouldSerializeKeyword::No>;
    using JumpNone  = Kind<Integer<Range{2,Range::infinity}>, Keyword::JumpNone>;

    std::variant<
        JumpStart,
        JumpEnd,
        JumpBoth,
        Start,
        End,
        JumpNone
    > value;

    bool operator==(const StepsEasingParameters&) const = default;
};
using StepsEasingFunction = FunctionNotation<CSSValueSteps, StepsEasingParameters>;

DEFINE_TYPE_WRAPPER_GET(StepsEasingParameters, value);

template<size_t I, typename T, typename K, auto shouldSerializeKeyword> const auto& get(const StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>& value)
{
    return value.steps;
}

template<typename T, typename K, auto shouldSerializeKeyword> struct Serialize<StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>> {
    void operator()(StringBuilder& builder, const StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>& value)
    {
        serializationForCSS(builder, value.steps);
        if constexpr (shouldSerializeKeyword == StepsEasingParameters::ShouldSerializeKeyword::Yes) {
            builder.append(", "_s);
            serializationForCSS(builder, value.keyword);
        }
    }
};

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::StepsEasingParameters, 1)

namespace std {

template<typename T, typename K, auto shouldSerializeKeyword> class tuple_size<WebCore::CSS::StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>> : public std::integral_constant<size_t, 1> { };
template<size_t I, typename T, typename K, auto shouldSerializeKeyword> class tuple_element<I, WebCore::CSS::StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>> {
public:
    using type = T;
};

}

template<typename T, typename K, auto shouldSerializeKeyword> inline constexpr bool WebCore::TreatAsTupleLike<WebCore::CSS::StepsEasingParameters::Kind<T, K, shouldSerializeKeyword>> = true;
