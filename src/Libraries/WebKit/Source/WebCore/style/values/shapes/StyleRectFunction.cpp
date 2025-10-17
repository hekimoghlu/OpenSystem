/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#include "config.h"
#include "StyleRectFunction.h"

#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

// MARK: - Conversion

auto ToStyle<CSS::Rect>::operator()(const CSS::Rect& value, const BuilderState& state) -> Inset
{
    // "An auto value makes the edge of the box coincide with the corresponding edge of the
    //  reference box: itâ€™s equivalent to 0% as the first (top) or fourth (left) value, and
    //  equivalent to 100% as the second (right) or third (bottom) value."
    //      (https://drafts.csswg.org/css-shapes-1/#funcdef-basic-shape-rect)

    // Conversion applies reflection to the trailing (right/bottom) edges to convert from rect()
    // form to inset() form. This means that all the `auto` values are converted to 0%.

    auto convertLeadingEdge = [&](const std::variant<CSS::LengthPercentage<>, CSS::Keyword::Auto>& edge) -> LengthPercentage<> {
        return WTF::switchOn(edge,
            [&](const CSS::LengthPercentage<>& value) -> LengthPercentage<> {
                return toStyle(value, state);
            },
            [&](const CSS::Keyword::Auto&) -> LengthPercentage<> {
                return { typename LengthPercentage<>::Percentage { 0 } };
            }
        );
    };

    auto convertTrailingEdge = [&](const std::variant<CSS::LengthPercentage<>, CSS::Keyword::Auto>& edge) -> LengthPercentage<> {
        return WTF::switchOn(edge,
            [&](const CSS::LengthPercentage<>& value) -> LengthPercentage<> {
                return reflect(toStyle(value, state));
            },
            [&](const CSS::Keyword::Auto&) -> LengthPercentage<> {
                return { typename LengthPercentage<>::Percentage { 0 } };
            }
        );
    };

    return {
        .insets = {
            convertLeadingEdge(value.edges.top()),
            convertTrailingEdge(value.edges.right()),
            convertTrailingEdge(value.edges.bottom()),
            convertLeadingEdge(value.edges.left()),
        },
        .radii = toStyle(value.radii, state)
    };
}

auto ToStyle<CSS::RectFunction>::operator()(const CSS::RectFunction& value, const BuilderState& state) -> InsetFunction
{
    return { toStyle(value.parameters, state) };
}

} // namespace Style
} // namespace WebCore
