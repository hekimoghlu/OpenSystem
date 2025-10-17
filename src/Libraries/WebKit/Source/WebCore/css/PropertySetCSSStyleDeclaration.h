/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "CSSParserContext.h"
#include "CSSStyleDeclaration.h"
#include "DeprecatedCSSOMValue.h"
#include "StyledElement.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CSSRule;
class CSSProperty;
class CSSValue;
class MutableStyleProperties;
class StyleSheetContents;
class StyledElement;

class PropertySetCSSStyleDeclaration : public CSSStyleDeclaration {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PropertySetCSSStyleDeclaration);
public:
    explicit PropertySetCSSStyleDeclaration(MutableStyleProperties& propertySet)
        : m_propertySet(&propertySet)
    { }

    void ref() const override;
    void deref() const override;

    StyleSheetContents* contextStyleSheet() const;

protected:
    enum class MutationType : uint8_t { NoChanges, StyleAttributeChanged, PropertyChanged };

    virtual CSSParserContext cssParserContext() const;

    MutableStyleProperties* m_propertySet;
    UncheckedKeyHashMap<CSSValue*, WeakPtr<DeprecatedCSSOMValue>> m_cssomValueWrappers;

private:
    CSSRule* parentRule() const override { return nullptr; }
    // FIXME: To implement.
    CSSRule* cssRules() const override { return nullptr; }
    unsigned length() const final;
    String item(unsigned index) const final;
    RefPtr<DeprecatedCSSOMValue> getPropertyCSSValue(const String& propertyName) final;
    String getPropertyValue(const String& propertyName) final;
    String getPropertyPriority(const String& propertyName) final;
    String getPropertyShorthand(const String& propertyName) final;
    bool isPropertyImplicit(const String& propertyName) final;
    ExceptionOr<void> setProperty(const String& propertyName, const String& value, const String& priority) final;
    ExceptionOr<String> removeProperty(const String& propertyName) final;
    String cssText() const final;
    ExceptionOr<void> setCssText(const String&) final;
    String getPropertyValueInternal(CSSPropertyID) final;
    ExceptionOr<void> setPropertyInternal(CSSPropertyID, const String& value, IsImportant) final;
    
    Ref<MutableStyleProperties> copyProperties() const final;
    bool isExposed(CSSPropertyID) const;

    RefPtr<DeprecatedCSSOMValue> wrapForDeprecatedCSSOM(CSSValue*);
    
    virtual bool willMutate() WARN_UNUSED_RETURN { return true; }
    virtual void didMutate(MutationType) { }
};

class StyleRuleCSSStyleDeclaration final : public PropertySetCSSStyleDeclaration, public RefCounted<StyleRuleCSSStyleDeclaration> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StyleRuleCSSStyleDeclaration);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<StyleRuleCSSStyleDeclaration> create(MutableStyleProperties& propertySet, CSSRule& parentRule)
    {
        return adoptRef(*new StyleRuleCSSStyleDeclaration(propertySet, parentRule));
    }
    virtual ~StyleRuleCSSStyleDeclaration();

    void clearParentRule() { m_parentRule = nullptr; }

    void reattach(MutableStyleProperties&);

private:
    StyleRuleCSSStyleDeclaration(MutableStyleProperties&, CSSRule&);

    CSSStyleSheet* parentStyleSheet() const final;

    CSSRule* parentRule() const final { return m_parentRule; }

    bool willMutate() final WARN_UNUSED_RETURN;
    void didMutate(MutationType) final;
    CSSParserContext cssParserContext() const final;

    StyleRuleType m_parentRuleType;
    CSSRule* m_parentRule;
};

class InlineCSSStyleDeclaration final : public PropertySetCSSStyleDeclaration {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InlineCSSStyleDeclaration);
public:
    InlineCSSStyleDeclaration(MutableStyleProperties& propertySet, StyledElement& parentElement)
        : PropertySetCSSStyleDeclaration(propertySet)
        , m_parentElement(parentElement)
    {
    }

private:
    CSSStyleSheet* parentStyleSheet() const final;
    StyledElement* parentElement() const final { return m_parentElement.get(); }

    bool willMutate() final WARN_UNUSED_RETURN;
    void didMutate(MutationType) final;
    CSSParserContext cssParserContext() const final;

    WeakPtr<StyledElement, WeakPtrImplWithEventTargetData> m_parentElement;
};

} // namespace WebCore
