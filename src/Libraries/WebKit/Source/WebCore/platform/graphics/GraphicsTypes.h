/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

#include "Color.h"
#include "FloatSize.h"
#include "WindRule.h"
#include <optional>
#include <wtf/Forward.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

enum class CompositeOperator : uint8_t {
    Clear,
    Copy,
    SourceOver,
    SourceIn,
    SourceOut,
    SourceAtop,
    DestinationOver,
    DestinationIn,
    DestinationOut,
    DestinationAtop,
    XOR,
    PlusDarker,
    PlusLighter,
    Difference
};

enum class BlendMode : uint8_t {
    Normal = 1, // Start with 1 to match SVG's blendmode enumeration.
    Multiply,
    Screen,
    Darken,
    Lighten,
    Overlay,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    PlusDarker,
    PlusLighter
};

struct CompositeMode {
    CompositeOperator operation;
    BlendMode blendMode;

    friend bool operator==(const CompositeMode&, const CompositeMode&) = default;
};

enum class DocumentMarkerLineStyleMode : uint8_t {
    TextCheckingDictationPhraseWithAlternatives,
    Spelling,
    Grammar,
    AutocorrectionReplacement,
    DictationAlternatives
};

struct DocumentMarkerLineStyle {
    DocumentMarkerLineStyleMode mode;
    Color color;
};

enum class GradientSpreadMethod : uint8_t {
    Pad,
    Reflect,
    Repeat
};

// InterpolationQuality::Default
// For ImagePaintingOptions, it means:
//  - Use context image interpolation quality.
// For GraphicsContext CG it means:
//  - If the CGImage has shouldInterpolate == true, use High
//  - Else use None
// For GraphicsContext Cairo it means:
//  - Use Medium
//
// FIXME: Remove InterpolationQuality::Default since it does not mean what it should
// obviously mean and because the CG context behavior is unusable in general case where
// the draw call sites cannot track where the native images are generated from.
enum class InterpolationQuality : uint8_t {
    Default,
    DoNotInterpolate,
    Low,
    Medium,
    High
};

enum class LineCap : uint8_t {
    Butt,
    Round,
    Square
};

enum class LineJoin : uint8_t {
    Miter,
    Round,
    Bevel
};

enum HorizontalAlignment {
    AlignLeft,
    AlignRight,
    AlignHCenter
};

enum class StrokeStyle : uint8_t {
    NoStroke,
    SolidStroke,
    DottedStroke,
    DashedStroke,
    DoubleStroke,
    WavyStroke,
};

enum class TextDrawingMode : uint8_t {
    Fill = 1 << 0,
    Stroke = 1 << 1,
};
using TextDrawingModeFlags = OptionSet<TextDrawingMode>;

enum TextBaseline {
    AlphabeticTextBaseline,
    TopTextBaseline,
    MiddleTextBaseline,
    BottomTextBaseline,
    IdeographicTextBaseline,
    HangingTextBaseline
};

enum TextAlign {
    StartTextAlign,
    EndTextAlign,
    LeftTextAlign,
    CenterTextAlign,
    RightTextAlign
};

String compositeOperatorName(WebCore::CompositeOperator, WebCore::BlendMode);
String blendModeName(WebCore::BlendMode);
bool parseBlendMode(const String&, WebCore::BlendMode&);
bool parseCompositeAndBlendOperator(const String&, WebCore::CompositeOperator&, WebCore::BlendMode&);

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, WebCore::BlendMode);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, WebCore::CompositeOperator);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, CompositeMode);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, GradientSpreadMethod);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, InterpolationQuality);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, LineCap);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, LineJoin);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, StrokeStyle);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, TextDrawingMode);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, WindRule);

} // namespace WebCore
