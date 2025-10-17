/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

#include "CSSPrimitiveValue.h"
#include "CSSProperty.h"
#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "Element.h"
#include "ElementData.h"

namespace WebCore {

class Attribute;
class ImmutableStyleProperties;
class MutableStyleProperties;
class PropertySetCSSStyleDeclaration;
class StyleProperties;
class StylePropertyMap;

class StyledElement : public Element {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StyledElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(StyledElement);
public:
    virtual ~StyledElement();

    void dirtyStyleAttribute();
    void invalidateStyleAttribute();

    const StyleProperties* inlineStyle() const { return elementData() ? elementData()->m_inlineStyle.get() : nullptr; }
    RefPtr<StyleProperties> protectedInlineStyle() const;
    
    WEBCORE_EXPORT bool setInlineStyleProperty(CSSPropertyID, CSSValueID identifier, IsImportant = IsImportant::No);
    bool setInlineStyleProperty(CSSPropertyID, CSSPropertyID identifier, IsImportant = IsImportant::No);
    WEBCORE_EXPORT bool setInlineStyleProperty(CSSPropertyID, double value, CSSUnitType, IsImportant = IsImportant::No);
    WEBCORE_EXPORT bool setInlineStyleProperty(CSSPropertyID, const String& value, IsImportant = IsImportant::No, bool* didFailParsing = nullptr);
    bool setInlineStyleCustomProperty(const AtomString& property, const String& value, IsImportant = IsImportant::No);
    bool setInlineStyleCustomProperty(Ref<CSSValue>&&, IsImportant = IsImportant::No);
    bool setInlineStyleProperty(CSSPropertyID, Ref<CSSValue>&&, IsImportant = IsImportant::No);
    WEBCORE_EXPORT bool removeInlineStyleProperty(CSSPropertyID);
    bool removeInlineStyleCustomProperty(const AtomString&);
    void removeAllInlineStyleProperties();

    void synchronizeStyleAttributeInternal() const { const_cast<StyledElement*>(this)->synchronizeStyleAttributeInternalImpl(); }
    
    WEBCORE_EXPORT CSSStyleDeclaration& cssomStyle();
    StylePropertyMap& ensureAttributeStyleMap();

    // https://html.spec.whatwg.org/#presentational-hints
    const ImmutableStyleProperties* presentationalHintStyle() const;
    virtual void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) { }
    virtual const MutableStyleProperties* additionalPresentationalHintStyle() const { return nullptr; }
    virtual void collectExtraStyleForPresentationalHints(MutableStyleProperties&) { }

protected:
    StyledElement(const QualifiedName& name, Document& document, OptionSet<TypeFlag> type)
        : Element(name, document, type)
    {
    }

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason = AttributeModificationReason::Directly) override;

    virtual bool hasPresentationalHintsForAttribute(const QualifiedName&) const { return false; }

    void addPropertyToPresentationalHintStyle(MutableStyleProperties&, CSSPropertyID, CSSValueID identifier);
    void addPropertyToPresentationalHintStyle(MutableStyleProperties&, CSSPropertyID, double value, CSSUnitType);
    void addPropertyToPresentationalHintStyle(MutableStyleProperties&, CSSPropertyID, const String& value);
    void addPropertyToPresentationalHintStyle(MutableStyleProperties&, CSSPropertyID, RefPtr<CSSValue>&&);

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const override;
    Attribute replaceURLsInAttributeValue(const Attribute&, const UncheckedKeyHashMap<String, String>&) const override;

private:
    void styleAttributeChanged(const AtomString& newStyleString, AttributeModificationReason);
    void synchronizeStyleAttributeInternalImpl();

    void inlineStyleChanged();
    PropertySetCSSStyleDeclaration* inlineStyleCSSOMWrapper();
    void setInlineStyleFromString(const AtomString&);
    MutableStyleProperties& ensureMutableInlineStyle();

    void rebuildPresentationalHintStyle();
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StyledElement)
    static bool isType(const WebCore::Node& node) { return node.isStyledElement(); }
SPECIALIZE_TYPE_TRAITS_END()
