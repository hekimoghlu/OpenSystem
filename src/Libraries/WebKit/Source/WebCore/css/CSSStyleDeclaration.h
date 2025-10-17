/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CheckedRef.h>

namespace WebCore {

class CSSRule;
class CSSStyleSheet;
class CSSValue;
class DeprecatedCSSOMValue;
class MutableStyleProperties;
class StyleProperties;
class StyledElement;

class CSSStyleDeclaration : public ScriptWrappable, public AbstractRefCountedAndCanMakeSingleThreadWeakPtr<CSSStyleDeclaration> {
    WTF_MAKE_NONCOPYABLE(CSSStyleDeclaration);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSStyleDeclaration);
public:
    virtual ~CSSStyleDeclaration() = default;

    virtual StyledElement* parentElement() const { return nullptr; }
    virtual CSSRule* parentRule() const = 0;
    virtual CSSRule* cssRules() const = 0;
    virtual String cssText() const = 0;
    virtual ExceptionOr<void> setCssText(const String&) = 0;
    virtual unsigned length() const = 0;
    virtual String item(unsigned index) const = 0;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    virtual RefPtr<DeprecatedCSSOMValue> getPropertyCSSValue(const String& propertyName) = 0;
    virtual String getPropertyValue(const String& propertyName) = 0;
    virtual String getPropertyPriority(const String& propertyName) = 0;
    virtual String getPropertyShorthand(const String& propertyName) = 0;
    virtual bool isPropertyImplicit(const String& propertyName) = 0;
    virtual ExceptionOr<void> setProperty(const String& propertyName, const String& value, const String& priority) = 0;
    virtual ExceptionOr<String> removeProperty(const String& propertyName) = 0;

    String cssFloat();
    ExceptionOr<void> setCssFloat(const String&);

    // CSSPropertyID versions of the CSSOM functions to support bindings and editing.
    // Use the non-virtual methods in the concrete subclasses when possible.
    virtual String getPropertyValueInternal(CSSPropertyID) = 0;
    virtual ExceptionOr<void> setPropertyInternal(CSSPropertyID, const String& value, IsImportant) = 0;

    virtual Ref<MutableStyleProperties> copyProperties() const = 0;

    virtual CSSStyleSheet* parentStyleSheet() const { return nullptr; }

    virtual const Settings* settings() const;

    // FIXME: It would be more efficient, by virtue of avoiding the text transformation and hash lookup currently
    // required in the implementation, if we could could smuggle the CSSPropertyID through the bindings, perhaps
    // by encoding it into the HashTableValue and then passing it together with the PropertyName.

    // Shared implementation for all properties that match https://drafts.csswg.org/cssom/#dom-cssstyledeclaration-camel_cased_attribute.
    String propertyValueForCamelCasedIDLAttribute(const AtomString&);
    ExceptionOr<void> setPropertyValueForCamelCasedIDLAttribute(const AtomString&, const String&);

    // Shared implementation for all properties that match https://drafts.csswg.org/cssom/#dom-cssstyledeclaration-webkit_cased_attribute.
    String propertyValueForWebKitCasedIDLAttribute(const AtomString&);
    ExceptionOr<void> setPropertyValueForWebKitCasedIDLAttribute(const AtomString&, const String&);

    // Shared implementation for all properties that match https://drafts.csswg.org/cssom/#dom-cssstyledeclaration-dashed_attribute.
    String propertyValueForDashedIDLAttribute(const AtomString&);
    ExceptionOr<void> setPropertyValueForDashedIDLAttribute(const AtomString&, const String&);

    // Shared implementation for all properties that match non-standard Epub-cased.
    String propertyValueForEpubCasedIDLAttribute(const AtomString&);
    ExceptionOr<void> setPropertyValueForEpubCasedIDLAttribute(const AtomString&, const String&);

    // FIXME: This needs to pass in a Settings& to work correctly.
    static CSSPropertyID getCSSPropertyIDFromJavaScriptPropertyName(const AtomString&);

protected:
    CSSStyleDeclaration() = default;
};

} // namespace WebCore
