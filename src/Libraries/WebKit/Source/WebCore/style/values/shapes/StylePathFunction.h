/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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

#include "CSSPathFunction.h"
#include "StyleFillRule.h"
#include "StylePathComputation.h"
#include "StylePrimitiveNumericTypes.h"
#include "StyleWindRuleComputation.h"

namespace WebCore {
namespace Style {

enum class PathConversion : bool { None, ForceAbsolute };

struct Path {
    struct Data {
        SVGPathByteStream byteStream;

        bool operator==(const Data&) const = default;
    };

    std::optional<FillRule> fillRule;
    Data data;

    float zoom { 1 };

    bool operator==(const Path&) const = default;
};
using PathFunction = FunctionNotation<CSSValuePath, Path>;

template<size_t I> const auto& get(const Path& value)
{
    if constexpr (!I)
        return value.fillRule;
    else if constexpr (I == 1)
        return value.data;
    else if constexpr (I == 2)
        return value.zoom;
}

template<> struct ToCSS<Path> { auto operator()(const Path&, const RenderStyle&) -> CSS::Path; };
template<> struct ToStyle<CSS::Path> { auto operator()(const CSS::Path&, const BuilderState&) -> Path; };

template<> struct ToCSS<Path::Data> { auto operator()(const Path::Data&, const RenderStyle&) -> CSS::Path::Data; };
template<> struct ToStyle<CSS::Path::Data> { auto operator()(const CSS::Path::Data&, const BuilderState&) -> Path::Data; };

// Non-standard parameters, `conversion` and `zoom`, are needed in some instances of Style <-> CSS conversions
// for Path, so additional "override" conversion operators are provided here.
auto overrideToCSS(const PathFunction&, const RenderStyle&, PathConversion) -> CSS::PathFunction;
auto overrideToStyle(const CSS::PathFunction&, const Style::BuilderState&, std::optional<float> zoom) -> PathFunction;

template<> struct PathComputation<Path> { WebCore::Path operator()(const Path&, const FloatRect&); };
template<> struct WindRuleComputation<Path> { WebCore::WindRule operator()(const Path&); };

template<> struct Blending<Path> {
    auto canBlend(const Path&, const Path&) -> bool;
    auto blend(const Path&, const Path&, const BlendingContext&) -> Path;
};

} // namespace Style
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::Style::Path, 2)
