/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
#include "CSSShapeFunction.h"

#include "CSSPrimitiveNumericTypes+Serialization.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

void Serialize<ToPosition>::operator()(StringBuilder& builder, const ToPosition& value)
{
    // <to-position> = to <position>

    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<ByCoordinatePair>::operator()(StringBuilder& builder, const ByCoordinatePair& value)
{
    // <by-coordinate-pair> = by <coordinate-pair>

    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<RelativeControlPoint>::operator()(StringBuilder& builder, const RelativeControlPoint& value)
{
    // <relative-control-point> = [<coordinate-pair> [from [start | end | origin]]?]
    // Specified https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    serializationForCSS(builder, value.offset);

    if (value.anchor) {
        builder.append(' ', nameLiteralForSerialization(CSSValueFrom), ' ');
        serializationForCSS(builder, *value.anchor);
    }
}

void Serialize<AbsoluteControlPoint>::operator()(StringBuilder& builder, const AbsoluteControlPoint& value)
{
    // <to-control-point> = [<position> | <relative-control-point>]
    // Specified https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    // Representation diverges from grammar due to overlap between <position> and <relative-control-point>.

    serializationForCSS(builder, value.offset);

    if (value.anchor) {
        builder.append(' ', nameLiteralForSerialization(CSSValueFrom), ' ');
        serializationForCSS(builder, *value.anchor);
    }
}

void Serialize<MoveCommand>::operator()(StringBuilder& builder, const MoveCommand& value)
{
    // <move-command> = move [to <position>] | [by <coordinate-pair>]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-move-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<LineCommand>::operator()(StringBuilder& builder, const LineCommand& value)
{
    // <line-command> = line [to <position>] | [by <coordinate-pair>]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-line-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<HLineCommand::To>::operator()(StringBuilder& builder, const HLineCommand::To& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<HLineCommand::By>::operator()(StringBuilder& builder, const HLineCommand::By& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<HLineCommand>::operator()(StringBuilder& builder, const HLineCommand& value)
{
    // <horizontal-line-command> = hline [ to [ <length-percentage> | left | center | right | x-start | x-end ] | by <length-percentage> ]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-hv-line-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2426552611

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<VLineCommand::To>::operator()(StringBuilder& builder, const VLineCommand::To& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<VLineCommand::By>::operator()(StringBuilder& builder, const VLineCommand::By& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);
}

void Serialize<VLineCommand>::operator()(StringBuilder& builder, const VLineCommand& value)
{
    // <vertical-line-command> = vline [ to [ <length-percentage> | top | center | bottom | y-start | y-end ] | by <length-percentage> ]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-hv-line-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2426552611

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<CurveCommand::To>::operator()(StringBuilder& builder, const CurveCommand::To& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);

    builder.append(' ', nameLiteralForSerialization(CSSValueWith), ' ');
    serializationForCSS(builder, value.controlPoint1);
    if (value.controlPoint2) {
        builder.append(" / "_s);
        serializationForCSS(builder, *value.controlPoint2);
    }
}

void Serialize<CurveCommand::By>::operator()(StringBuilder& builder, const CurveCommand::By& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);

    builder.append(' ', nameLiteralForSerialization(CSSValueWith), ' ');
    serializationForCSS(builder, value.controlPoint1);
    if (value.controlPoint2) {
        builder.append(" / "_s);
        serializationForCSS(builder, *value.controlPoint2);
    }
}

void Serialize<CurveCommand>::operator()(StringBuilder& builder, const CurveCommand& value)
{
    // <curve-command> = curve [to <position> with <to-control-point> [/ <to-control-point>]?]
    //                       | [by <coordinate-pair> with <relative-control-point> [/ <relative-control-point>]?]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-curve-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<SmoothCommand::To>::operator()(StringBuilder& builder, const SmoothCommand::To& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);

    if (value.controlPoint) {
        builder.append(' ', nameLiteralForSerialization(CSSValueWith), ' ');
        serializationForCSS(builder, *value.controlPoint);
    }
}

void Serialize<SmoothCommand::By>::operator()(StringBuilder& builder, const SmoothCommand::By& value)
{
    builder.append(nameLiteralForSerialization(value.affinity.value), ' ');
    serializationForCSS(builder, value.offset);

    if (value.controlPoint) {
        builder.append(' ', nameLiteralForSerialization(CSSValueWith), ' ');
        serializationForCSS(builder, *value.controlPoint);
    }
}

void Serialize<SmoothCommand>::operator()(StringBuilder& builder, const SmoothCommand& value)
{
    // <smooth-command> = smooth [to <position> [with <to-control-point>]?]
    //                         | [by <coordinate-pair> [with <relative-control-point>]?]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-smooth-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);
}

void Serialize<ArcCommand>::operator()(StringBuilder& builder, const ArcCommand& value)
{
    // <arc-command> = arc [to <position>] | [by <coordinate-pair>] of <length-percentage>{1,2} [<arc-sweep>? || <arc-size>? || [rotate <angle>]?]
    // https://drafts.csswg.org/css-shapes-2/#typedef-shape-arc-command
    // Modified by https://github.com/w3c/csswg-drafts/issues/10649#issuecomment-2412816773

    builder.append(nameLiteralForSerialization(value.name), ' ');
    serializationForCSS(builder, value.toBy);

    builder.append(' ', nameLiteralForSerialization(CSSValueOf), ' ');
    if (value.size.width() == value.size.height())
        serializationForCSS(builder, value.size.width());
    else
        serializationForCSS(builder, value.size);

    if (!std::holds_alternative<CSS::Keyword::Ccw>(value.arcSweep)) {
        builder.append(' ');
        serializationForCSS(builder, value.arcSweep);
    }

    if (!std::holds_alternative<CSS::Keyword::Small>(value.arcSize)) {
        builder.append(' ');
        serializationForCSS(builder, value.arcSize);
    }

    if (value.rotation != 0_css_deg) {
        builder.append(' ', nameLiteralForSerialization(CSSValueRotate), ' ');
        serializationForCSS(builder, value.rotation);
    }
}

void Serialize<Shape>::operator()(StringBuilder& builder, const Shape& value)
{
    // shape() = shape( <'fill-rule'>? from <coordinate-pair>, <shape-command>#)

    if (value.fillRule && !std::holds_alternative<Keyword::Nonzero>(*value.fillRule)) {
        serializationForCSS(builder, *value.fillRule);
        builder.append(' ');
    }

    builder.append(nameLiteralForSerialization(CSSValueFrom), ' ');
    serializationForCSS(builder, value.startingPoint);
    builder.append(", "_s);
    serializationForCSS(builder, value.commands);
}

} // namespace CSS
} // namespace WebCore
