/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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

#include "CSSProperty.h"
#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "WritingDirection.h"
#include <wtf/RefCounted.h>
#include <wtf/TriState.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSStyleDeclaration;
class CSSComputedStyleDeclaration;
class CSSPrimitiveValue;
class CSSValue;
class ComputedStyleExtractor;
class Document;
class Element;
class HTMLElement;
class MutableStyleProperties;
class Node;
class Position;
class QualifiedName;
class RenderStyle;
class StyleProperties;
class StyledElement;
class VisibleSelection;

enum class TextDecorationChange { None, Add, Remove };

// FIXME: "Keep" should be "Resolve" instead and resolve all generic font family names.
enum class StandardFontFamilySerializationMode : uint8_t { Keep, Strip };

class EditingStyle : public RefCounted<EditingStyle> {
public:

    enum PropertiesToInclude { AllProperties, OnlyEditingInheritableProperties, EditingPropertiesInEffect };

    enum ShouldPreserveWritingDirection { PreserveWritingDirection, DoNotPreserveWritingDirection };
    enum ShouldExtractMatchingStyle { ExtractMatchingStyle, DoNotExtractMatchingStyle };
    static float NoFontDelta;

    static Ref<EditingStyle> create()
    {
        return adoptRef(*new EditingStyle);
    }

    static Ref<EditingStyle> create(Node* node, PropertiesToInclude propertiesToInclude = OnlyEditingInheritableProperties)
    {
        return adoptRef(*new EditingStyle(node, propertiesToInclude));
    }

    static Ref<EditingStyle> create(const Position& position, PropertiesToInclude propertiesToInclude = OnlyEditingInheritableProperties)
    {
        return adoptRef(*new EditingStyle(position, propertiesToInclude));
    }

    static Ref<EditingStyle> create(const StyleProperties* style)
    {
        return adoptRef(*new EditingStyle(style));
    }

    static Ref<EditingStyle> create(const CSSStyleDeclaration* style)
    {
        return adoptRef(*new EditingStyle(style));
    }

    static Ref<EditingStyle> create(CSSPropertyID propertyID, const String& value)
    {
        return adoptRef(*new EditingStyle(propertyID, value));
    }

    static Ref<EditingStyle> create(CSSPropertyID propertyID, CSSValueID value)
    {
        return adoptRef(*new EditingStyle(propertyID, value));
    }

    WEBCORE_EXPORT ~EditingStyle();

    MutableStyleProperties* style() { return m_mutableStyle.get(); }
    RefPtr<MutableStyleProperties> protectedStyle();
    Ref<MutableStyleProperties> styleWithResolvedTextDecorations() const;
    std::optional<WritingDirection> textDirection() const;
    bool isEmpty() const;
    void setStyle(RefPtr<MutableStyleProperties>&&);
    void overrideWithStyle(const StyleProperties&);
    void overrideTypingStyleAt(const EditingStyle&, const Position&);
    void clear();
    Ref<EditingStyle> copy() const;
    Ref<EditingStyle> extractAndRemoveBlockProperties();
    Ref<EditingStyle> extractAndRemoveTextDirection();
    void removeBlockProperties();
    void removeStyleAddedByNode(Node*);
    void removeStyleConflictingWithStyleOfNode(Node&);
    template<typename T> void removeEquivalentProperties(T&);
    void collapseTextDecorationProperties();
    enum ShouldIgnoreTextOnlyProperties { IgnoreTextOnlyProperties, DoNotIgnoreTextOnlyProperties };
    TriState triStateOfStyle(EditingStyle*) const;
    TriState triStateOfStyle(const VisibleSelection&) const;
    bool conflictsWithInlineStyleOfElement(StyledElement& element) const { return conflictsWithInlineStyleOfElement(element, nullptr, nullptr); }
    bool conflictsWithInlineStyleOfElement(StyledElement& element, RefPtr<MutableStyleProperties>& newInlineStyle, EditingStyle* extractedStyle) const
    {
        return conflictsWithInlineStyleOfElement(element, &newInlineStyle, extractedStyle);
    }
    bool conflictsWithImplicitStyleOfElement(HTMLElement&, EditingStyle* extractedStyle = nullptr, ShouldExtractMatchingStyle = DoNotExtractMatchingStyle) const;
    bool conflictsWithImplicitStyleOfAttributes(HTMLElement&) const;
    bool extractConflictingImplicitStyleOfAttributes(HTMLElement&, ShouldPreserveWritingDirection, EditingStyle* extractedStyle, Vector<QualifiedName>& conflictingAttributes, ShouldExtractMatchingStyle) const;
    bool styleIsPresentInComputedStyleOfNode(Node&) const;

    static bool elementIsStyledSpanOrHTMLEquivalent(const HTMLElement&);

    void prepareToApplyAt(const Position&, ShouldPreserveWritingDirection = DoNotPreserveWritingDirection);
    void mergeTypingStyle(Document&);
    enum CSSPropertyOverrideMode { OverrideValues, DoNotOverrideValues };
    void mergeInlineStyleOfElement(StyledElement&, CSSPropertyOverrideMode, PropertiesToInclude = AllProperties);
    static Ref<EditingStyle> wrappingStyleForSerialization(Node& context, bool shouldAnnotate, StandardFontFamilySerializationMode);
    void mergeStyleFromRules(StyledElement&);
    void mergeStyleFromRulesForSerialization(StyledElement&, StandardFontFamilySerializationMode);
    void removeStyleFromRulesAndContext(StyledElement&, Node* context);
    void removePropertiesInElementDefaultStyle(Element&);
    void forceInline();
    void addDisplayContents();
    bool convertPositionStyle();
    bool isFloating();
    int legacyFontSize(Document&) const;

    float fontSizeDelta() const { return m_fontSizeDelta; }
    bool hasFontSizeDelta() const { return m_fontSizeDelta != NoFontDelta; }
    bool shouldUseFixedDefaultFontSize() const { return m_shouldUseFixedDefaultFontSize; }
    
    void setUnderlineChange(TextDecorationChange change) { m_underlineChange = static_cast<unsigned>(change); }
    TextDecorationChange underlineChange() const { return static_cast<TextDecorationChange>(m_underlineChange); }
    void setStrikeThroughChange(TextDecorationChange change) { m_strikeThroughChange = static_cast<unsigned>(change); }
    TextDecorationChange strikeThroughChange() const { return static_cast<TextDecorationChange>(m_strikeThroughChange); }

    WEBCORE_EXPORT bool hasStyle(CSSPropertyID, const String& value);
    WEBCORE_EXPORT static RefPtr<EditingStyle> styleAtSelectionStart(const VisibleSelection&, bool shouldUseBackgroundColorInEffect = false);
    static WritingDirection textDirectionForSelection(const VisibleSelection&, EditingStyle* typingStyle, bool& hasNestedOrMultipleEmbeddings);
    static bool isEmbedOrIsolate(CSSValueID unicodeBidi) { return unicodeBidi == CSSValueID::CSSValueIsolate || unicodeBidi == CSSValueID::CSSValueWebkitIsolate || unicodeBidi == CSSValueID::CSSValueEmbed; }

    Ref<EditingStyle> inverseTransformColorIfNeeded(Element&);

private:
    EditingStyle();
    EditingStyle(Node*, PropertiesToInclude);
    EditingStyle(const Position&, PropertiesToInclude);
    WEBCORE_EXPORT explicit EditingStyle(const CSSStyleDeclaration*);
    explicit EditingStyle(const StyleProperties*);
    EditingStyle(CSSPropertyID, const String& value);
    EditingStyle(CSSPropertyID, CSSValueID);
    void init(Node*, PropertiesToInclude);
    void removeTextFillAndStrokeColorsIfNeeded(const RenderStyle*);
    void setProperty(CSSPropertyID, const String& value, IsImportant = IsImportant::No);
    void extractFontSizeDelta();
    template<typename T> TriState triStateOfStyle(T& styleToCompare, ShouldIgnoreTextOnlyProperties) const;
    bool conflictsWithInlineStyleOfElement(StyledElement&, RefPtr<MutableStyleProperties>* newInlineStyle, EditingStyle* extractedStyle) const;
    void mergeInlineAndImplicitStyleOfElement(StyledElement&, CSSPropertyOverrideMode, PropertiesToInclude, StandardFontFamilySerializationMode);
    void mergeStyle(const StyleProperties*, CSSPropertyOverrideMode);

    RefPtr<MutableStyleProperties> m_mutableStyle;
    unsigned m_shouldUseFixedDefaultFontSize : 1;
    unsigned m_underlineChange : 2;
    unsigned m_strikeThroughChange : 2;
    float m_fontSizeDelta = NoFontDelta;

    friend class HTMLElementEquivalent;
    friend class HTMLAttributeEquivalent;
    friend class HTMLTextDecorationEquivalent;
    friend class HTMLFontWeightEquivalent;
};

class StyleChange {
public:
    StyleChange() = default;
    StyleChange(EditingStyle*, const Position&);
    ~StyleChange();

    const MutableStyleProperties* cssStyle() const { return m_cssStyle.get(); }
    bool applyBold() const { return m_applyBold; }
    bool applyItalic() const { return m_applyItalic; }
    bool applyUnderline() const { return m_applyUnderline; }
    bool applyLineThrough() const { return m_applyLineThrough; }
    bool applySubscript() const { return m_applySubscript; }
    bool applySuperscript() const { return m_applySuperscript; }
    bool applyFontColor() const { return m_applyFontColor.length() > 0; }
    bool applyFontFace() const { return m_applyFontFace.length() > 0; }
    bool applyFontSize() const { return m_applyFontSize.length() > 0; }

    const AtomString& fontColor() { return m_applyFontColor; }
    const AtomString& fontFace() { return m_applyFontFace; }
    const AtomString& fontSize() { return m_applyFontSize; }

    bool operator==(const StyleChange&);

private:
    void extractTextStyles(Document&, MutableStyleProperties&, bool shouldUseFixedFontDefaultSize);

    RefPtr<MutableStyleProperties> m_cssStyle;
    bool m_applyBold = false;
    bool m_applyItalic = false;
    bool m_applyUnderline = false;
    bool m_applyLineThrough = false;
    bool m_applySubscript = false;
    bool m_applySuperscript = false;
    AtomString m_applyFontColor;
    AtomString m_applyFontFace;
    AtomString m_applyFontSize;
};

} // namespace WebCore
