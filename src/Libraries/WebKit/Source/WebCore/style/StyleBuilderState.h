/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

#include "CSSToLengthConversionData.h"
#include "CSSToStyleMap.h"
#include "CascadeLevel.h"
#include "PropertyCascade.h"
#include "RuleSet.h"
#include "SelectorChecker.h"
#include "StyleForVisitedLink.h"
#include <wtf/BitSet.h>

namespace WebCore {

class FilterOperations;
class FontCascadeDescription;
class RenderStyle;
class StyleImage;
class StyleResolver;

namespace Calculation {
class RandomKeyMap;
}

namespace CSS {
struct AppleColorFilterProperty;
struct FilterProperty;
}

namespace Style {

class Builder;
class BuilderState;
struct Color;

void maybeUpdateFontForLetterSpacing(BuilderState&, CSSValue&);

enum class ApplyValueType : uint8_t { Value, Initial, Inherit };

struct BuilderContext {
    Ref<const Document> document;
    const RenderStyle& parentStyle;
    const RenderStyle* rootElementStyle = nullptr;
    RefPtr<const Element> element = nullptr;
};

class BuilderState {
public:
    BuilderState(Builder&, RenderStyle&, BuilderContext&&);

    Builder& builder() { return m_builder; }

    RenderStyle& style() { return m_style; }
    const RenderStyle& style() const { return m_style; }

    const RenderStyle& parentStyle() const { return m_context.parentStyle; }
    const RenderStyle* rootElementStyle() const { return m_context.rootElementStyle; }

    const Document& document() const { return m_context.document.get(); }
    const Element* element() const { return m_context.element.get(); }

    inline void setFontDescription(FontCascadeDescription&&);
    void setFontSize(FontCascadeDescription&, float size);
    inline void setZoom(float);
    inline void setUsedZoom(float);
    inline void setWritingMode(StyleWritingMode);
    inline void setTextOrientation(TextOrientation);

    bool fontDirty() const { return m_fontDirty; }
    void setFontDirty() { m_fontDirty = true; }

    inline const FontCascadeDescription& fontDescription();
    inline const FontCascadeDescription& parentFontDescription();

    bool applyPropertyToRegularStyle() const { return m_linkMatch != SelectorChecker::MatchVisited; }
    bool applyPropertyToVisitedLinkStyle() const { return m_linkMatch != SelectorChecker::MatchLink; }

    bool useSVGZoomRules() const;
    bool useSVGZoomRulesForLength() const;
    ScopeOrdinal styleScopeOrdinal() const { return m_currentProperty->styleScopeOrdinal; }

    RefPtr<StyleImage> createStyleImage(const CSSValue&) const;
    FilterOperations createFilterOperations(const CSS::FilterProperty&) const;
    FilterOperations createFilterOperations(const CSSValue&) const;
    FilterOperations createAppleColorFilterOperations(const CSS::AppleColorFilterProperty&) const;
    FilterOperations createAppleColorFilterOperations(const CSSValue&) const;
    Color createStyleColor(const CSSValue&, ForVisitedLink = ForVisitedLink::No) const;

    const Vector<AtomString>& registeredContentAttributes() const { return m_registeredContentAttributes; }
    void registerContentAttribute(const AtomString& attributeLocalName);

    const CSSToLengthConversionData& cssToLengthConversionData() const { return m_cssToLengthConversionData; }
    CSSToStyleMap& styleMap() { return m_styleMap; }

    void setIsBuildingKeyframeStyle() { m_isBuildingKeyframeStyle = true; }

    bool isAuthorOrigin() const
    {
        return m_currentProperty && m_currentProperty->cascadeLevel == CascadeLevel::Author;
    }

    CSSPropertyID cssPropertyID() const;

    bool isCurrentPropertyInvalidAtComputedValueTime() const;
    void setCurrentPropertyInvalidAtComputedValueTime();

    Ref<Calculation::RandomKeyMap> randomKeyMap(bool perElement) const;

private:
    // See the comment in maybeUpdateFontForLetterSpacing() about why this needs to be a friend.
    friend void maybeUpdateFontForLetterSpacing(BuilderState&, CSSValue&);
    friend class Builder;

    void adjustStyleForInterCharacterRuby();

    void updateFont();
#if ENABLE(TEXT_AUTOSIZING)
    void updateFontForTextSizeAdjust();
#endif
    void updateFontForZoomChange();
    void updateFontForGenericFamilyChange();
    void updateFontForOrientationChange();

    Builder& m_builder;

    CSSToStyleMap m_styleMap;

    RenderStyle& m_style;
    const BuilderContext m_context;

    const CSSToLengthConversionData m_cssToLengthConversionData;

    UncheckedKeyHashSet<AtomString> m_appliedCustomProperties;
    UncheckedKeyHashSet<AtomString> m_inProgressCustomProperties;
    UncheckedKeyHashSet<AtomString> m_inCycleCustomProperties;
    WTF::BitSet<numCSSProperties> m_inProgressProperties;
    WTF::BitSet<numCSSProperties> m_invalidAtComputedValueTimeProperties;

    const PropertyCascade::Property* m_currentProperty { nullptr };
    SelectorChecker::LinkMatchMask m_linkMatch { };

    bool m_fontDirty { false };
    Vector<AtomString> m_registeredContentAttributes;

    bool m_isBuildingKeyframeStyle { false };
};

} // namespace Style
} // namespace WebCore
