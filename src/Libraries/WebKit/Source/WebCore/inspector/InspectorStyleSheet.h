/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#include "CSSPropertySourceData.h"
#include "CSSStyleDeclaration.h"
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/HashMap.h>
#include <wtf/JSONValues.h>
#include <wtf/Vector.h>

namespace WebCore {

class CSSRuleList;
class CSSSelector;
class CSSStyleDeclaration;
class CSSStyleRule;
class CSSStyleSheet;
class Document;
class Element;
class InspectorPageAgent;
class InspectorStyleSheet;
class ParsedStyleSheet;

class InspectorCSSId {
public:
    InspectorCSSId() = default;

    explicit InspectorCSSId(const JSON::Object& value)
        : m_styleSheetId(value.getString("styleSheetId"_s))
    {
        if (!m_styleSheetId)
            return;

        if (auto ordinal = value.getInteger("ordinal"_s))
            m_ordinal = *ordinal;
        else
            m_styleSheetId = String();
    }

    InspectorCSSId(const String& styleSheetId, unsigned ordinal)
        : m_styleSheetId(styleSheetId)
        , m_ordinal(ordinal)
    {
    }

    bool isEmpty() const { return m_styleSheetId.isEmpty(); }

    const String& styleSheetId() const { return m_styleSheetId; }
    unsigned ordinal() const { return m_ordinal; }

    // ID type is either Inspector::Protocol::CSS::CSSStyleId or Inspector::Protocol::CSS::CSSRuleId.
    template<typename ID>
    RefPtr<ID> asProtocolValue() const
    {
        if (isEmpty())
            return nullptr;

        return ID::create()
            .setStyleSheetId(m_styleSheetId)
            .setOrdinal(m_ordinal)
            .release();
    }

private:
    String m_styleSheetId;
    unsigned m_ordinal = {0};
};

struct InspectorStyleProperty {
    InspectorStyleProperty()
        : hasSource(false)
        , disabled(false)
    {
    }

    InspectorStyleProperty(CSSPropertySourceData sourceData, bool hasSource, bool disabled)
        : sourceData(sourceData)
        , hasSource(hasSource)
        , disabled(disabled)
    {
    }

    void setRawTextFromStyleDeclaration(const String& styleDeclaration)
    {
        unsigned start = sourceData.range.start;
        unsigned end = sourceData.range.end;
        ASSERT_WITH_SECURITY_IMPLICATION(start < end);
        ASSERT(end <= styleDeclaration.length());
        rawText = styleDeclaration.substring(start, end - start);
    }

    bool hasRawText() const { return !rawText.isEmpty(); }

    CSSPropertySourceData sourceData;
    bool hasSource;
    bool disabled;
    String rawText;
};

class InspectorStyle final : public RefCounted<InspectorStyle> {
public:
    static Ref<InspectorStyle> create(const InspectorCSSId& styleId, Ref<CSSStyleDeclaration>&&, InspectorStyleSheet* parentStyleSheet);
    ~InspectorStyle();

    CSSStyleDeclaration& cssStyle() const { return m_style.get(); }
    Ref<Inspector::Protocol::CSS::CSSStyle> buildObjectForStyle();
    Ref<JSON::ArrayOf<Inspector::Protocol::CSS::CSSComputedStyleProperty>> buildArrayForComputedStyle();

    ExceptionOr<String> text();

private:
    InspectorStyle(const InspectorCSSId& styleId, Ref<CSSStyleDeclaration>&&, InspectorStyleSheet* parentStyleSheet);

    Vector<InspectorStyleProperty> collectProperties(bool includeAll);
    Ref<Inspector::Protocol::CSS::CSSStyle> styleWithProperties();
    RefPtr<CSSRuleSourceData> extractSourceData() const;
    String shorthandValue(const String& shorthandProperty) const;
    String shorthandPriority(const String& shorthandProperty) const;
    Vector<String> longhandProperties(const String& shorthandProperty) const;

    InspectorCSSId m_styleId;
    Ref<CSSStyleDeclaration> m_style;
    InspectorStyleSheet* m_parentStyleSheet;
};

class InspectorStyleSheet : public RefCounted<InspectorStyleSheet> {
public:
    class Listener {
    public:
        Listener() = default;
        virtual ~Listener() = default;
        virtual void styleSheetChanged(InspectorStyleSheet*) = 0;
    };

    using StyleDeclarationOrCSSRule = std::variant<CSSStyleDeclaration*, CSSRule*>;

    static Ref<InspectorStyleSheet> create(InspectorPageAgent*, const String& id, RefPtr<CSSStyleSheet>&& pageStyleSheet, Inspector::Protocol::CSS::StyleSheetOrigin, const String& documentURL, Listener*);
    static String styleSheetURL(CSSStyleSheet* pageStyleSheet);

    virtual ~InspectorStyleSheet();

    String id() const { return m_id; }
    String finalURL() const;
    CSSStyleSheet* pageStyleSheet() const { return m_pageStyleSheet.get(); }
    void reparseStyleSheet(const String&);
    ExceptionOr<void> setText(const String&);
    ExceptionOr<String> ruleHeaderText(const InspectorCSSId&);
    ExceptionOr<void> setRuleHeaderText(const InspectorCSSId&, const String& selector);
    ExceptionOr<CSSStyleRule*> addRule(const String& selector);
    ExceptionOr<void> deleteRule(const InspectorCSSId&);
    CSSRule* ruleForId(const InspectorCSSId&) const;
    RefPtr<Inspector::Protocol::CSS::CSSStyleSheetBody> buildObjectForStyleSheet();
    RefPtr<Inspector::Protocol::CSS::CSSStyleSheetHeader> buildObjectForStyleSheetInfo();
    RefPtr<Inspector::Protocol::CSS::CSSRule> buildObjectForRule(CSSStyleRule*);
    Ref<Inspector::Protocol::CSS::CSSStyle> buildObjectForStyle(CSSStyleDeclaration*);
    RefPtr<Inspector::Protocol::CSS::Grouping> buildObjectForGrouping(CSSRule*);

    virtual ExceptionOr<void> setRuleStyleText(const InspectorCSSId&, const String& newStyleDeclarationText, String* outOldStyleDeclarationText, const String* newRuleText, String* outOldRuleText);

    virtual ExceptionOr<String> text();
    virtual CSSStyleDeclaration* styleForId(const InspectorCSSId&) const;
    void fireStyleSheetChanged();

    InspectorCSSId ruleOrStyleId(StyleDeclarationOrCSSRule) const;

protected:
    InspectorStyleSheet(InspectorPageAgent*, const String& id, RefPtr<CSSStyleSheet>&& pageStyleSheet, Inspector::Protocol::CSS::StyleSheetOrigin, const String& documentURL, Listener*);

    bool canBind() const { return m_origin != Inspector::Protocol::CSS::StyleSheetOrigin::UserAgent && m_origin != Inspector::Protocol::CSS::StyleSheetOrigin::User; }
    virtual Document* ownerDocument() const;
    virtual RefPtr<CSSRuleSourceData> ruleSourceDataFor(CSSStyleDeclaration*) const;
    virtual RefPtr<CSSRuleSourceData> ruleSourceDataFor(CSSRule*) const;
    virtual unsigned ruleIndexByStyle(StyleDeclarationOrCSSRule, bool combineSplitRules = false) const;

    virtual bool ensureParsedDataReady();
    virtual RefPtr<InspectorStyle> inspectorStyleForId(const InspectorCSSId&);

    // Also accessed by friend class InspectorStyle.
    virtual Vector<size_t> lineEndings() const;

private:
    friend class InspectorStyle;

    static void collectFlatRules(RefPtr<CSSRuleList>&&, Vector<RefPtr<CSSRule>>* result);
    bool ensureText();
    bool ensureSourceData();
    void ensureFlatRules() const;
    bool originalStyleSheetText(String* result) const;
    bool resourceStyleSheetText(String* result) const;
    bool inlineStyleSheetText(String* result) const;
    bool extensionStyleSheetText(String* result) const;
    bool styleSheetTextFromCSSRuleSerialization(String* result) const;
    Ref<JSON::ArrayOf<Inspector::Protocol::CSS::Grouping>> buildArrayForGroupings(CSSRule&);
    Ref<JSON::ArrayOf<Inspector::Protocol::CSS::CSSRule>> buildArrayForRuleList(CSSRuleList*);
    Ref<Inspector::Protocol::CSS::CSSSelector> buildObjectForSelector(const CSSSelector*);
    Ref<Inspector::Protocol::CSS::SelectorList> buildObjectForSelectorList(CSSStyleRule*, int& endingLine);

    Vector<Ref<CSSStyleRule>> cssStyleRulesSplitFromSameRule(CSSStyleRule&);
    Vector<const CSSSelector*> selectorsForCSSStyleRule(CSSStyleRule&);

    InspectorPageAgent* m_pageAgent;
    String m_id;
    RefPtr<CSSStyleSheet> m_pageStyleSheet;
    Inspector::Protocol::CSS::StyleSheetOrigin m_origin;
    String m_documentURL;
    ParsedStyleSheet* m_parsedStyleSheet;
    mutable Vector<RefPtr<CSSRule>> m_flatRules;
    Listener* m_listener;
};

class InspectorStyleSheetForInlineStyle final : public InspectorStyleSheet {
public:
    static Ref<InspectorStyleSheetForInlineStyle> create(InspectorPageAgent*, const String& id, Ref<StyledElement>&&, Inspector::Protocol::CSS::StyleSheetOrigin, Listener*);

    void didModifyElementAttribute();
    ExceptionOr<String> text() final;
    CSSStyleDeclaration* styleForId(const InspectorCSSId& id) const final { ASSERT_UNUSED(id, !id.ordinal()); return &inlineStyle(); }
    ExceptionOr<void> setRuleStyleText(const InspectorCSSId&, const String& newStyleDeclarationText, String* outOldStyleDeclarationText, const String* newRuleText, String* outOldRuleText);

private:
    InspectorStyleSheetForInlineStyle(InspectorPageAgent*, const String& id, Ref<StyledElement>&&, Inspector::Protocol::CSS::StyleSheetOrigin, Listener*);

    Document* ownerDocument() const final;
    RefPtr<CSSRuleSourceData> ruleSourceDataFor(CSSStyleDeclaration* style) const final { ASSERT_UNUSED(style, style == &inlineStyle()); return m_ruleSourceData; }
    RefPtr<CSSRuleSourceData> ruleSourceDataFor(CSSRule* rule) const final { ASSERT_UNUSED(rule, rule); return m_ruleSourceData; }
    unsigned ruleIndexByStyle(StyleDeclarationOrCSSRule, bool) const final { return 0; }
    bool ensureParsedDataReady() final;
    RefPtr<InspectorStyle> inspectorStyleForId(const InspectorCSSId&) final;

    // Also accessed by friend class InspectorStyle.
    Vector<size_t> lineEndings() const final;

    CSSStyleDeclaration& inlineStyle() const;
    const String& elementStyleText() const;
    Ref<CSSRuleSourceData> ruleSourceData() const;

    Ref<StyledElement> m_element;
    RefPtr<CSSRuleSourceData> m_ruleSourceData;
    RefPtr<InspectorStyle> m_inspectorStyle;

    // Contains "style" attribute value.
    mutable String m_styleText;
    mutable bool m_isStyleTextValid;
};

} // namespace WebCore
