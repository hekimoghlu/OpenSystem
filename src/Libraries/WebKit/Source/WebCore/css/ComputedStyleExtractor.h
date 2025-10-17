/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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

#include "PseudoElementIdentifier.h"
#include <span>
#include <wtf/RefPtr.h>

namespace WebCore {

namespace Style {
struct Color;
}

class Animation;
class CSSColorValue;
class CSSFunctionValue;
class CSSPrimitiveValue;
class CSSValue;
class CSSValueList;
class Element;
class FilterOperations;
class MutableStyleProperties;
class Node;
class RenderElement;
class RenderStyle;
class ShadowData;
class StylePropertyShorthand;
class TransformOperation;
class TransformationMatrix;

struct Length;
struct PropertyValue;

enum CSSPropertyID : uint16_t;
enum CSSValueID : uint16_t;

enum class PseudoId : uint32_t;
enum class SVGPaintType : uint8_t;

using CSSValueListBuilder = Vector<Ref<CSSValue>, 4>;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ComputedStyleExtractor);
class ComputedStyleExtractor {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(ComputedStyleExtractor);
public:
    ComputedStyleExtractor(Node*, bool allowVisitedStyle = false);
    ComputedStyleExtractor(Node*, bool allowVisitedStyle, const std::optional<Style::PseudoElementIdentifier>&);
    ComputedStyleExtractor(Element*, bool allowVisitedStyle = false);
    ComputedStyleExtractor(Element*, bool allowVisitedStyle, const std::optional<Style::PseudoElementIdentifier>&);

    enum class UpdateLayout : bool { No, Yes };
    enum class PropertyValueType : bool { Resolved, Computed };
    bool hasProperty(CSSPropertyID) const;
    RefPtr<CSSValue> propertyValue(CSSPropertyID, UpdateLayout = UpdateLayout::Yes, PropertyValueType = PropertyValueType::Resolved) const;
    RefPtr<CSSValue> valueForPropertyInStyle(const RenderStyle&, CSSPropertyID, RenderElement* = nullptr, PropertyValueType = PropertyValueType::Resolved) const;
    String customPropertyText(const AtomString& propertyName) const;
    RefPtr<CSSValue> customPropertyValue(const AtomString& propertyName) const;

    // Helper methods for HTML editing.
    Ref<MutableStyleProperties> copyProperties(std::span<const CSSPropertyID>) const;
    Ref<MutableStyleProperties> copyProperties() const;
    RefPtr<CSSPrimitiveValue> getFontSizeCSSValuePreferringKeyword() const;
    bool useFixedFontDefaultSize() const;
    bool propertyMatches(CSSPropertyID, const CSSValue*) const;
    bool propertyMatches(CSSPropertyID, CSSValueID) const;

    static Ref<CSSValue> cssValueForFilter(const RenderStyle&, const FilterOperations&);
    static Ref<CSSValue> cssValueForAppleColorFilter(const RenderStyle&, const FilterOperations&);

    static Ref<CSSColorValue> currentColorOrValidColor(const RenderStyle&, const Style::Color&);
    static Ref<CSSFunctionValue> matrixTransformValue(const TransformationMatrix&, const RenderStyle&);
    static Ref<CSSPrimitiveValue> zoomAdjustedPixelValueForLength(const Length&, const RenderStyle&);

    static bool updateStyleIfNeededForProperty(Element&, CSSPropertyID);

private:
    // The renderer we should use for resolving layout-dependent properties.
    RenderElement* styledRenderer() const;

    RefPtr<CSSValue> svgPropertyValue(CSSPropertyID) const;
    Ref<CSSValue> adjustSVGPaint(SVGPaintType, const String& url, Ref<CSSValue> color) const;

    Ref<CSSValueList> getCSSPropertyValuesForShorthandProperties(const StylePropertyShorthand&) const;
    RefPtr<CSSValueList> getCSSPropertyValuesFor2SidesShorthand(const StylePropertyShorthand&) const;
    RefPtr<CSSValueList> getCSSPropertyValuesFor4SidesShorthand(const StylePropertyShorthand&) const;

    size_t getLayerCount(CSSPropertyID) const;
    Ref<CSSValue> getFillLayerPropertyShorthandValue(CSSPropertyID, const StylePropertyShorthand& propertiesBeforeSlashSeparator, const StylePropertyShorthand& propertiesAfterSlashSeparator, CSSPropertyID lastLayerProperty) const;
    Ref<CSSValue> getBackgroundShorthandValue() const;
    Ref<CSSValue> getMaskShorthandValue() const;
    Ref<CSSValueList> getCSSPropertyValuesForGridShorthand(const StylePropertyShorthand&) const;
    Ref<CSSValue> fontVariantShorthandValue() const;
    RefPtr<CSSValue> textWrapShorthandValue(const RenderStyle&) const;
    RefPtr<CSSValue> whiteSpaceShorthandValue(const RenderStyle&) const;
    RefPtr<CSSValue> textBoxShorthandValue(const RenderStyle&) const;
    RefPtr<CSSValue> lineClampShorthandValue(const RenderStyle&) const;

    RefPtr<Element> m_element;
    std::optional<Style::PseudoElementIdentifier> m_pseudoElementIdentifier;
    bool m_allowVisitedStyle;
};

RefPtr<CSSFunctionValue> transformOperationAsCSSValue(const TransformOperation&, const RenderStyle&);

} // namespace WebCore
