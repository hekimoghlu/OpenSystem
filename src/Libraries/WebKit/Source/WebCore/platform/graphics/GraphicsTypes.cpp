/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "GraphicsTypes.h"

#include "AlphaPremultiplication.h"
#include <wtf/Assertions.h>
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static constexpr std::array compositeOperatorNames {
    "clear"_s,
    "copy"_s,
    "source-over"_s,
    "source-in"_s,
    "source-out"_s,
    "source-atop"_s,
    "destination-over"_s,
    "destination-in"_s,
    "destination-out"_s,
    "destination-atop"_s,
    "xor"_s,
    "darker"_s,
    "lighter"_s,
    "difference"_s
};

static constexpr std::array blendOperatorNames {
    "normal"_s,
    "multiply"_s,
    "screen"_s,
    "darken"_s,
    "lighten"_s,
    "overlay"_s,
    "color-dodge"_s,
    "color-burn"_s,
    "hard-light"_s,
    "soft-light"_s,
    "difference"_s,
    "exclusion"_s,
    "hue"_s,
    "saturation"_s,
    "color"_s,
    "luminosity"_s,
    "plus-darker"_s,
    "plus-lighter"_s
};
const uint8_t numCompositeOperatorNames = std::size(compositeOperatorNames);
const uint8_t numBlendOperatorNames = std::size(blendOperatorNames);

bool parseBlendMode(const String& s, BlendMode& blendMode)
{
    for (unsigned i = 0; i < numBlendOperatorNames; i++) {
        if (s == blendOperatorNames[i]) {
            blendMode = static_cast<BlendMode>(i + static_cast<unsigned>(BlendMode::Normal));
            return true;
        }
    }
    
    return false;
}

bool parseCompositeAndBlendOperator(const String& s, CompositeOperator& op, BlendMode& blendOp)
{
    for (int i = 0; i < numCompositeOperatorNames; i++) {
        if (s == compositeOperatorNames[i]) {
            op = static_cast<CompositeOperator>(i);
            blendOp = BlendMode::Normal;
            return true;
        }
    }
    
    if (parseBlendMode(s, blendOp)) {
        // For now, blending will always assume source-over. This will be fixed in the future
        op = CompositeOperator::SourceOver;
        return true;
    }
    
    return false;
}

// FIXME: when we support blend modes in combination with compositing other than source-over
// this routine needs to be updated.
String compositeOperatorName(CompositeOperator op, BlendMode blendOp)
{
    ASSERT(op >= CompositeOperator::Clear);
    ASSERT(static_cast<uint8_t>(op) < numCompositeOperatorNames);
    ASSERT(blendOp >= BlendMode::Normal);
    ASSERT(static_cast<uint8_t>(blendOp) <= numBlendOperatorNames);
    if (blendOp > BlendMode::Normal)
        return blendOperatorNames[static_cast<unsigned>(blendOp) - static_cast<unsigned>(BlendMode::Normal)];
    return compositeOperatorNames[static_cast<unsigned>(op)];
}

String blendModeName(BlendMode blendOp)
{
    ASSERT(blendOp >= BlendMode::Normal);
    ASSERT(blendOp <= BlendMode::PlusLighter);
    return blendOperatorNames[static_cast<unsigned>(blendOp) - static_cast<unsigned>(BlendMode::Normal)];
}

TextStream& operator<<(TextStream& ts, CompositeOperator op)
{
    return ts << compositeOperatorName(op, BlendMode::Normal);
}

TextStream& operator<<(TextStream& ts, BlendMode blendMode)
{
    return ts << blendModeName(blendMode);
}

TextStream& operator<<(TextStream& ts, CompositeMode compositeMode)
{
    ts.dumpProperty("composite-operation", compositeMode.operation);
    ts.dumpProperty("blend-mode", compositeMode.blendMode);
    return ts;
}

TextStream& operator<<(TextStream& ts, GradientSpreadMethod spreadMethod)
{
    switch (spreadMethod) {
    case GradientSpreadMethod::Pad:
        ts << "pad";
        break;
    case GradientSpreadMethod::Reflect:
        ts << "reflect";
        break;
    case GradientSpreadMethod::Repeat:
        ts << "repeat";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, InterpolationQuality interpolationQuality)
{
    switch (interpolationQuality) {
    case InterpolationQuality::Default:
        ts << "default";
        break;
    case InterpolationQuality::DoNotInterpolate:
        ts << "do-not-interpolate";
        break;
    case InterpolationQuality::Low:
        ts << "low";
        break;
    case InterpolationQuality::Medium:
        ts << "medium";
        break;
    case InterpolationQuality::High:
        ts << "high";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, WindRule rule)
{
    switch (rule) {
    case WindRule::NonZero:
        ts << "NON-ZERO";
        break;
    case WindRule::EvenOdd:
        ts << "EVEN-ODD";
        break;
    }

    return ts;
}

TextStream& operator<<(TextStream& ts, LineCap capStyle)
{
    switch (capStyle) {
    case LineCap::Butt:
        ts << "BUTT";
        break;
    case LineCap::Round:
        ts << "ROUND";
        break;
    case LineCap::Square:
        ts << "SQUARE";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, LineJoin joinStyle)
{
    switch (joinStyle) {
    case LineJoin::Miter:
        ts << "MITER";
        break;
    case LineJoin::Round:
        ts << "ROUND";
        break;
    case LineJoin::Bevel:
        ts << "BEVEL";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, StrokeStyle strokeStyle)
{
    switch (strokeStyle) {
    case StrokeStyle::NoStroke:
        ts << "no-stroke";
        break;
    case StrokeStyle::SolidStroke:
        ts << "solid-stroke";
        break;
    case StrokeStyle::DottedStroke:
        ts << "dotted-stroke";
        break;
    case StrokeStyle::DashedStroke:
        ts << "dashed-stroke";
        break;
    case StrokeStyle::DoubleStroke:
        ts << "double-stroke";
        break;
    case StrokeStyle::WavyStroke:
        ts << "wavy-stroke";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, TextDrawingMode textDrawingMode)
{
    switch (textDrawingMode) {
    case TextDrawingMode::Fill:
        ts << "fill";
        break;
    case TextDrawingMode::Stroke:
        ts << "stroke";
        break;
    }
    return ts;
}

} // namespace WebCore
