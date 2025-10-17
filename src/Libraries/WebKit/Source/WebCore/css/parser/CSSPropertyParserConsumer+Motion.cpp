/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "CSSPropertyParserConsumer+Motion.h"

#include "CSSOffsetRotateValue.h"
#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+Angle.h"
#include "CSSPropertyParserConsumer+AngleDefinitions.h"
#include "CSSPropertyParserConsumer+Ident.h"
#include "CSSPropertyParserConsumer+MetaConsumer.h"
#include "CSSPropertyParserConsumer+Position.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSPropertyParserConsumer+Shapes.h"
#include "CSSPropertyParserConsumer+URL.h"
#include "CSSPropertyParsing.h"
#include "CSSRayValue.h"
#include "CSSValueKeywords.h"
#include "CSSValueList.h"
#include <wtf/SortedArrayMap.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

static RefPtr<CSSValue> consumeRayFunction(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // ray( <angle> && <ray-size>? && contain? && [at <position>]? )
    // <ray-size> = closest-side | closest-corner | farthest-side | farthest-corner | sides
    // https://drafts.fxtf.org/motion-1/#ray-function

    static constexpr std::pair<CSSValueID, CSS::RaySize> sizeMappings[] {
        { CSSValueClosestSide, CSS::RaySize { CSS::Keyword::ClosestSide { } } },
        { CSSValueClosestCorner, CSS::RaySize { CSS::Keyword::ClosestCorner { } } },
        { CSSValueFarthestSide, CSS::RaySize { CSS::Keyword::FarthestSide { } } },
        { CSSValueFarthestCorner, CSS::RaySize { CSS::Keyword::FarthestCorner { } } },
        { CSSValueSides, CSS::RaySize { CSS::Keyword::Sides { } } },
    };
    static constexpr SortedArrayMap sizeMap { sizeMappings };

    if (range.peek().type() != FunctionToken || range.peek().functionId() != CSSValueRay)
        return { };

    auto args = consumeFunction(range);

    std::optional<CSS::Angle<>> angle;
    std::optional<CSS::RaySize> size;
    std::optional<CSS::Keyword::Contain> contain;
    std::optional<CSS::Position> position;

    auto consumeAngle = [&] -> bool {
        if (angle)
            return false;
        angle = MetaConsumer<CSS::Angle<>>::consume(args, context, { }, { .parserMode = context.mode });
        return angle.has_value();
    };
    auto consumeSize = [&] -> bool {
        if (size)
            return false;
        size = consumeIdentUsingMapping(args, sizeMap);
        return size.has_value();
    };
    auto consumeContain = [&] -> bool {
        if (contain || !consumeIdentRaw<CSSValueContain>(args).has_value())
            return false;
        contain = CSS::Keyword::Contain { };
        return contain.has_value();
    };
    auto consumeAtPosition = [&] -> bool {
        if (position || !consumeIdentRaw<CSSValueAt>(args).has_value())
            return false;
        position = consumePositionUnresolved(args, context);
        return position.has_value();
    };

    while (!args.atEnd()) {
        if (consumeAngle() || consumeSize() || consumeContain() || consumeAtPosition())
            continue;
        return { };
    }

    // The <angle> argument is the only one that is required.
    if (!angle)
        return { };

    return CSSRayValue::create(
        CSS::RayFunction {
            .parameters = CSS::Ray {
                WTFMove(*angle),
                size.value_or(CSS::RaySize { CSS::Keyword::ClosestSide { } }),
                WTFMove(contain),
                WTFMove(position)
            }
        }
    );
}

RefPtr<CSSValue> consumeOffsetPath(CSSParserTokenRange& range, const CSSParserContext& context)
{
    // <'offset-path'> = none | <offset-path> || <coord-box>
    //
    // NOTE: The sub-production, <offset-path> (without the quotation marks) is distinct and defined as:
    //    <offset-path> = <ray()> | <url> | <basic-shape>
    //
    // So, this expands out to a grammar of:
    //
    // <'offset-path'> = none | [ <ray()> | <url> | <basic-shape> || <coord-box> ]
    //
    // which is almost the same as <'clip-path'> above, with the following differences:
    //
    // 1. <'clip-path'> does not support `ray()`.
    // 2. <'clip-path'> does not allow a `box` to be provided with `<url>`.
    // 3. <'clip-path'> specifies `<geometry-box>` rather than `<coord-box>`.
    //
    // https://drafts.fxtf.org/motion-1/#propdef-offset-path

    if (range.peek().id() == CSSValueNone)
        return consumeIdent(range);

    // FIXME: It should be possible to consume both a <url> and <coord-box>.
    if (auto url = consumeURL(range))
        return url;

    RefPtr<CSSValue> shapeOrRay;
    RefPtr<CSSValue> box;

    auto consumeRay = [&]() -> bool {
        if (shapeOrRay)
            return false;
        shapeOrRay = consumeRayFunction(range, context);
        return !!shapeOrRay;
    };
    auto consumeShape = [&]() -> bool {
        if (shapeOrRay)
            return false;
        shapeOrRay = consumeBasicShape(range, context, PathParsingOption::RejectPathFillRule);
        return !!shapeOrRay;
    };
    auto consumeBox = [&]() -> bool {
        if (box)
            return false;
        // FIXME: The Motion Path spec calls for this to be a <coord-box>, not a <geometry-box>, the difference being that the former does not contain "margin-box" as a valid term.
        // However, the spec also has a few examples using "margin-box", so there seems to be some abiguity to be resolved. See: https://github.com/w3c/fxtf-drafts/issues/481.
        box = CSSPropertyParsing::consumeGeometryBox(range);
        return !!box;
    };

    while (!range.atEnd()) {
        if (consumeRay() || consumeShape() || consumeBox())
            continue;
        break;
    }

    bool hasShapeOrRay = !!shapeOrRay;

    CSSValueListBuilder list;
    if (shapeOrRay)
        list.append(shapeOrRay.releaseNonNull());

    // Default value is border-box.
    if (box && (box->valueID() != CSSValueBorderBox || !hasShapeOrRay))
        list.append(box.releaseNonNull());

    if (list.isEmpty())
        return nullptr;

    return CSSValueList::createSpaceSeparated(WTFMove(list));
}

RefPtr<CSSValue> consumeOffsetRotate(CSSParserTokenRange& range, const CSSParserContext& context)
{
    auto rangeCopy = range;

    // Attempt to parse the first token as the modifier (auto / reverse keyword). If
    // successful, parse the second token as the angle. If not, try to parse the other
    // way around.
    auto modifier = consumeIdent<CSSValueAuto, CSSValueReverse>(rangeCopy);
    auto angle = consumeAngle(rangeCopy, context);
    if (!modifier)
        modifier = consumeIdent<CSSValueAuto, CSSValueReverse>(rangeCopy);
    if (!angle && !modifier)
        return nullptr;

    range = rangeCopy;
    return CSSOffsetRotateValue::create(WTFMove(modifier), WTFMove(angle));
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
