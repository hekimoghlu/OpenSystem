/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include "HTMLOptGroupElement.h"

#include "Document.h"
#include "ElementAncestorIteratorInlines.h"
#include "ElementIterator.h"
#include "HTMLNames.h"
#include "HTMLOptionElement.h"
#include "HTMLSelectElement.h"
#include "PseudoClassChangeInvalidation.h"
#include "RenderMenuList.h"
#include "NodeRenderStyle.h"
#include "StyleResolver.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLOptGroupElement);

using namespace HTMLNames;

inline HTMLOptGroupElement::HTMLOptGroupElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(optgroupTag));
}

Ref<HTMLOptGroupElement> HTMLOptGroupElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLOptGroupElement(tagName, document));
}

bool HTMLOptGroupElement::isDisabledFormControl() const
{
    return m_isDisabled;
}

bool HTMLOptGroupElement::isFocusable() const
{
    RefPtr select = ownerSelectElement();
    if (select && select->usesMenuList())
        return false;
    return HTMLElement::isFocusable();
}

const AtomString& HTMLOptGroupElement::formControlType() const
{
    static MainThreadNeverDestroyed<const AtomString> optgroup("optgroup"_s);
    return optgroup;
}

void HTMLOptGroupElement::childrenChanged(const ChildChange& change)
{
    bool isRelevant = change.affectsElements == ChildChange::AffectsElements::Yes;
    RefPtr select = isRelevant ? ownerSelectElement() : nullptr;
    if (!isRelevant || !select) {
        HTMLElement::childrenChanged(change);
        return;
    }

    auto selectOptionIfNecessaryScope = select->optionToSelectFromChildChangeScope(change, this);

    recalcSelectOptions();
    HTMLElement::childrenChanged(change);
}

void HTMLOptGroupElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
    recalcSelectOptions();

    if (name == disabledAttr) {
        bool newDisabled = !newValue.isNull();
        if (m_isDisabled != newDisabled) {
            Style::PseudoClassChangeInvalidation disabledInvalidation(*this, { { CSSSelector::PseudoClass::Disabled, newDisabled }, { CSSSelector::PseudoClass::Enabled, !newDisabled } });

            Vector<Style::PseudoClassChangeInvalidation> optionInvalidation;
            for (auto& descendant : descendantsOfType<HTMLOptionElement>(*this))
                optionInvalidation.append({ descendant, { { CSSSelector::PseudoClass::Disabled, newDisabled }, { CSSSelector::PseudoClass::Enabled, !newDisabled } } });

            m_isDisabled = newDisabled;
        }
    }
}

void HTMLOptGroupElement::recalcSelectOptions()
{
    if (RefPtr selectElement = ownerSelectElement()) {
        selectElement->setRecalcListItems();
        selectElement->updateValidity();
    }
}

String HTMLOptGroupElement::groupLabelText() const
{
    String itemText = document().displayStringModifiedByEncoding(attributeWithoutSynchronization(labelAttr));
    
    // In WinIE, leading and trailing whitespace is ignored in options and optgroups. We match this behavior.
    itemText = itemText.trim(deprecatedIsSpaceOrNewline);
    // We want to collapse our whitespace too.  This will match other browsers.
    itemText = itemText.simplifyWhiteSpace(deprecatedIsSpaceOrNewline);
        
    return itemText;
}
    
HTMLSelectElement* HTMLOptGroupElement::ownerSelectElement() const
{
    return dynamicDowncast<HTMLSelectElement>(parentNode());
}

bool HTMLOptGroupElement::accessKeyAction(bool)
{
    RefPtr select = ownerSelectElement();
    // send to the parent to bring focus to the list box
    if (select && !select->focused())
        return select->accessKeyAction(false);
    return false;
}

} // namespace
