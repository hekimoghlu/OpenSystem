/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#include "StylePosition.h"

#include "CalculationCategory.h"
#include "CalculationTree.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

auto ToStyle<CSS::TwoComponentPositionHorizontal>::operator()(const CSS::TwoComponentPositionHorizontal& value, const BuilderState& state) -> TwoComponentPositionHorizontal
{
    return WTF::switchOn(value.offset,
        [&](CSS::Keyword::Left) {
            return TwoComponentPositionHorizontal { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 0 } } };
        },
        [&](CSS::Keyword::Right)  {
            return TwoComponentPositionHorizontal { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 100 } } };
        },
        [&](CSS::Keyword::Center)  {
            return TwoComponentPositionHorizontal { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 50 } } };
        },
        [&](const CSS::LengthPercentage<>& value) {
            return TwoComponentPositionHorizontal { .offset = toStyle(value, state) };
        }
    );
}

auto ToStyle<CSS::TwoComponentPositionVertical>::operator()(const CSS::TwoComponentPositionVertical& value, const BuilderState& state) -> TwoComponentPositionVertical
{
    return WTF::switchOn(value.offset,
        [&](CSS::Keyword::Top) {
            return TwoComponentPositionVertical { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 0 } } };
        },
        [&](CSS::Keyword::Bottom) {
            return TwoComponentPositionVertical { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 100 } } };
        },
        [&](CSS::Keyword::Center) {
            return TwoComponentPositionVertical { .offset = LengthPercentage<> { typename LengthPercentage<>::Percentage { 50 } } };
        },
        [&](const CSS::LengthPercentage<>& value) {
            return TwoComponentPositionVertical { .offset = toStyle(value, state) };
        }
    );
}

auto ToCSS<Position>::operator()(const Position& value, const RenderStyle& style) -> CSS::Position
{
    return CSS::TwoComponentPosition { { toCSS(value.x(), style) }, { toCSS(value.y(), style) } };
}

auto ToStyle<CSS::Position>::operator()(const CSS::Position& position, const BuilderState& state) -> Position
{
    return WTF::switchOn(position,
        [&](const CSS::TwoComponentPosition& twoComponent) {
            return Position {
                toStyle(get<0>(twoComponent), state),
                toStyle(get<1>(twoComponent), state)
            };
        },
        [&](const CSS::FourComponentPosition& fourComponent) {
            auto horizontal = WTF::switchOn(get<0>(get<0>(fourComponent)),
                [&](CSS::Keyword::Left) {
                    return toStyle(get<1>(get<0>(fourComponent)), state);
                },
                [&](CSS::Keyword::Right) {
                    return reflect(toStyle(get<1>(get<0>(fourComponent)), state));
                }
            );
            auto vertical = WTF::switchOn(get<0>(get<1>(fourComponent)),
                [&](CSS::Keyword::Top) {
                    return toStyle(get<1>(get<1>(fourComponent)), state);
                },
                [&](CSS::Keyword::Bottom) {
                    return reflect(toStyle(get<1>(get<1>(fourComponent)), state));
                }
            );
            return Position { WTFMove(horizontal), WTFMove(vertical) };
        }
    );
}

// MARK: - Evaluation

auto Evaluation<Position>::operator()(const Position& position, FloatSize referenceBox) -> FloatPoint
{
    return evaluate(position.value, referenceBox);
}

} // namespace CSS
} // namespace WebCore
