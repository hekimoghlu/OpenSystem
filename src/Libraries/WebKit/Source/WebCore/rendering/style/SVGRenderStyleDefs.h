/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

#include "Length.h"
#include "SVGLengthValue.h"
#include "ShadowData.h"
#include "StyleColor.h"
#include "StylePathData.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class CSSValue;
class CSSValueList;
class SVGPaint;

enum class SVGPaintType : uint8_t {
    RGBColor,
    None,
    CurrentColor,
    URINone,
    URICurrentColor,
    URIRGBColor,
    URI
};

enum class BaselineShift : uint8_t {
    Baseline,
    Sub,
    Super,
    Length
};

enum class TextAnchor : uint8_t {
    Start,
    Middle,
    End
};

enum class ColorInterpolation : uint8_t {
    Auto,
    SRGB,
    LinearRGB
};

enum class ColorRendering : uint8_t {
    Auto,
    OptimizeSpeed,
    OptimizeQuality
};

enum class ShapeRendering : uint8_t {
    Auto,
    OptimizeSpeed,
    CrispEdges,
    GeometricPrecision
};

enum class GlyphOrientation : uint8_t {
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
    Auto
};

enum class AlignmentBaseline : uint8_t {
    Baseline,
    BeforeEdge,
    TextBeforeEdge,
    Middle,
    Central,
    AfterEdge,
    TextAfterEdge,
    Ideographic,
    Alphabetic,
    Hanging,
    Mathematical
};

enum class DominantBaseline : uint8_t {
    Auto,
    UseScript,
    NoChange,
    ResetSize,
    Ideographic,
    Alphabetic,
    Hanging,
    Mathematical,
    Central,
    Middle,
    TextAfterEdge,
    TextBeforeEdge
};

enum class VectorEffect : uint8_t {
    None,
    NonScalingStroke
};

enum class BufferedRendering : uint8_t {
    Auto,
    Dynamic,
    Static
};

enum class MaskType : uint8_t {
    Luminance,
    Alpha
};

// Inherited/Non-Inherited Style Datastructures
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleFillData);
class StyleFillData : public RefCounted<StyleFillData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleFillData);
public:
    static Ref<StyleFillData> create() { return adoptRef(*new StyleFillData); }
    Ref<StyleFillData> copy() const;

    bool operator==(const StyleFillData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleFillData&) const;
#endif

    float opacity;
    Style::Color paintColor;
    Style::Color visitedLinkPaintColor;
    String paintUri;
    String visitedLinkPaintUri;
    SVGPaintType paintType;
    SVGPaintType visitedLinkPaintType;

private:
    StyleFillData();
    StyleFillData(const StyleFillData&);
};

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleStrokeData);
class StyleStrokeData : public RefCounted<StyleStrokeData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleStrokeData);
public:
    static Ref<StyleStrokeData> create() { return adoptRef(*new StyleStrokeData); }
    Ref<StyleStrokeData> copy() const;

    bool operator==(const StyleStrokeData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleStrokeData&) const;
#endif

    float opacity;

    Style::Color paintColor;
    Style::Color visitedLinkPaintColor;

    String paintUri;
    String visitedLinkPaintUri;

    Length dashOffset;
    Vector<SVGLengthValue> dashArray;

    SVGPaintType paintType;
    SVGPaintType visitedLinkPaintType;

private:
    StyleStrokeData();
    StyleStrokeData(const StyleStrokeData&);
};

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleStopData);
class StyleStopData : public RefCounted<StyleStopData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleStopData);
public:
    static Ref<StyleStopData> create() { return adoptRef(*new StyleStopData); }
    Ref<StyleStopData> copy() const;

    bool operator==(const StyleStopData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleStopData&) const;
#endif

    float opacity;
    Style::Color color;

private:
    StyleStopData();
    StyleStopData(const StyleStopData&);
};

// Note: the rule for this class is, *no inheritance* of these props
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleMiscData);
class StyleMiscData : public RefCounted<StyleMiscData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleMiscData);
public:
    static Ref<StyleMiscData> create() { return adoptRef(*new StyleMiscData); }
    Ref<StyleMiscData> copy() const;

    bool operator==(const StyleMiscData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleMiscData&) const;
#endif

    float floodOpacity;
    Style::Color floodColor;
    Style::Color lightingColor;

    SVGLengthValue baselineShiftValue;

private:
    StyleMiscData();
    StyleMiscData(const StyleMiscData&);
};

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleShadowSVGData);
class StyleShadowSVGData : public RefCounted<StyleShadowSVGData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleShadowSVGData);
public:
    static Ref<StyleShadowSVGData> create() { return adoptRef(*new StyleShadowSVGData); }
    Ref<StyleShadowSVGData> copy() const;

    bool operator==(const StyleShadowSVGData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleShadowSVGData&) const;
#endif

    std::unique_ptr<ShadowData> shadow;

private:
    StyleShadowSVGData();
    StyleShadowSVGData(const StyleShadowSVGData&);
};

// Inherited resources
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleInheritedResourceData);
class StyleInheritedResourceData : public RefCounted<StyleInheritedResourceData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleInheritedResourceData);
public:
    static Ref<StyleInheritedResourceData> create() { return adoptRef(*new StyleInheritedResourceData); }
    Ref<StyleInheritedResourceData> copy() const;

    bool operator==(const StyleInheritedResourceData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleInheritedResourceData&) const;
#endif

    String markerStart;
    String markerMid;
    String markerEnd;

private:
    StyleInheritedResourceData();
    StyleInheritedResourceData(const StyleInheritedResourceData&);
};

// Positioning and sizing properties.
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleLayoutData);
class StyleLayoutData : public RefCounted<StyleLayoutData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleLayoutData);
public:
    static Ref<StyleLayoutData> create() { return adoptRef(*new StyleLayoutData); }
    Ref<StyleLayoutData> copy() const;

    bool operator==(const StyleLayoutData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleLayoutData&) const;
#endif

    Length cx;
    Length cy;
    Length r;
    Length rx;
    Length ry;
    Length x;
    Length y;
    RefPtr<StylePathData> d;

private:
    StyleLayoutData();
    StyleLayoutData(const StyleLayoutData&);
};


WTF::TextStream& operator<<(WTF::TextStream&, AlignmentBaseline);
WTF::TextStream& operator<<(WTF::TextStream&, BaselineShift);
WTF::TextStream& operator<<(WTF::TextStream&, BufferedRendering);
WTF::TextStream& operator<<(WTF::TextStream&, ColorInterpolation);
WTF::TextStream& operator<<(WTF::TextStream&, ColorRendering);
WTF::TextStream& operator<<(WTF::TextStream&, DominantBaseline);
WTF::TextStream& operator<<(WTF::TextStream&, GlyphOrientation);
WTF::TextStream& operator<<(WTF::TextStream&, MaskType);
WTF::TextStream& operator<<(WTF::TextStream&, SVGPaintType);
WTF::TextStream& operator<<(WTF::TextStream&, ShapeRendering);
WTF::TextStream& operator<<(WTF::TextStream&, TextAnchor);
WTF::TextStream& operator<<(WTF::TextStream&, VectorEffect);

WTF::TextStream& operator<<(WTF::TextStream&, const StyleFillData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleStrokeData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleStopData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleMiscData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleShadowSVGData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleInheritedResourceData&);
WTF::TextStream& operator<<(WTF::TextStream&, const StyleLayoutData&);

} // namespace WebCore
