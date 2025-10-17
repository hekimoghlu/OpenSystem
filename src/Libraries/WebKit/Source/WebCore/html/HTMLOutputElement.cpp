/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#include "HTMLOutputElement.h"

#include "DOMTokenList.h"
#include "Document.h"
#include "ElementInlines.h"
#include "HTMLFormElement.h"
#include "HTMLNames.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLOutputElement);

inline HTMLOutputElement::HTMLOutputElement(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
    : HTMLFormControlElement(tagName, document, form)
{
}

HTMLOutputElement::~HTMLOutputElement() = default;

Ref<HTMLOutputElement> HTMLOutputElement::create(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
{
    return adoptRef(*new HTMLOutputElement(tagName, document, form));
}

Ref<HTMLOutputElement> HTMLOutputElement::create(Document& document)
{
    return create(HTMLNames::outputTag, document, nullptr);
}

const AtomString& HTMLOutputElement::formControlType() const
{
    static MainThreadNeverDestroyed<const AtomString> output("output"_s);
    return output;
}

bool HTMLOutputElement::supportsFocus() const
{
    return HTMLElement::supportsFocus();
}

void HTMLOutputElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == HTMLNames::forAttr && m_forTokens)
        m_forTokens->associatedAttributeValueChanged();
    HTMLFormControlElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void HTMLOutputElement::reset()
{
    setInteractedWithSinceLastFormSubmitEvent(false);
    stringReplaceAll(defaultValue());
    m_defaultValueOverride = { };
}

String HTMLOutputElement::value() const
{
    return textContent();
}

void HTMLOutputElement::setValue(String&& value)
{
    m_defaultValueOverride = defaultValue();
    stringReplaceAll(WTFMove(value));
}

String HTMLOutputElement::defaultValue() const
{
    return m_defaultValueOverride.isNull() ? textContent() : m_defaultValueOverride;
}

void HTMLOutputElement::setDefaultValue(String&& value)
{
    if (m_defaultValueOverride.isNull())
        stringReplaceAll(WTFMove(value));
    else
        m_defaultValueOverride = WTFMove(value);
}

DOMTokenList& HTMLOutputElement::htmlFor()
{
    if (!m_forTokens)
        m_forTokens = makeUniqueWithoutRefCountedCheck<DOMTokenList>(*this, HTMLNames::forAttr);
    return *m_forTokens;
}

} // namespace
