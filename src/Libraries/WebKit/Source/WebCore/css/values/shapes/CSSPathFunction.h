/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#include "CSSFillRule.h"
#include "CSSPrimitiveNumericTypes.h"
#include "SVGPathByteStream.h"

namespace WebCore {
namespace CSS {

// <path()> = path( <'fill-rule'>? , <string> )
// https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-path
struct Path {
    struct Data {
        SVGPathByteStream byteStream;

        bool operator==(const Data&) const = default;
    };

    std::optional<FillRule> fillRule;
    Data data;

    bool operator==(const Path&) const = default;
};
using PathFunction = FunctionNotation<CSSValuePath, Path>;

template<size_t I> const auto& get(const Path& value)
{
    if constexpr (!I)
        return value.fillRule;
    else if constexpr (I == 1)
        return value.data;
}

template<> struct Serialize<Path> { void operator()(StringBuilder&, const Path&); };

template<> struct ComputedStyleDependenciesCollector<Path::Data> { void operator()(ComputedStyleDependencies&, const Path::Data&); };
template<> struct CSSValueChildrenVisitor<Path::Data> { IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const Path::Data&); };

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::Path, 2)
