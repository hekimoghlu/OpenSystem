/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#include "HTMLFieldSetElement.h"

#include "ElementChildIteratorInlines.h"
#include "GenericCachedHTMLCollection.h"
#include "HTMLFormControlsCollection.h"
#include "HTMLLegendElement.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "NodeRareData.h"
#include "PseudoClassChangeInvalidation.h"
#include "RenderElement.h"
#include "ScriptDisallowedScope.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFieldSetElement);

using namespace HTMLNames;

inline HTMLFieldSetElement::HTMLFieldSetElement(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
    : HTMLFormControlElement(tagName, document, form)
{
    ASSERT(hasTagName(fieldsetTag));
}

HTMLFieldSetElement::~HTMLFieldSetElement()
{
    if (hasDisabledAttribute())
        document().removeDisabledFieldsetElement();
}

Ref<HTMLFieldSetElement> HTMLFieldSetElement::create(const QualifiedName& tagName, Document& document, HTMLFormElement* form)
{
    return adoptRef(*new HTMLFieldSetElement(tagName, document, form));
}

bool HTMLFieldSetElement::isDisabledFormControl() const
{
    // The fieldset element itself should never be considered disabled, it is
    // only supposed to affect its descendants:
    // https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#concept-fe-disabled
    return false;
}

// https://html.spec.whatwg.org/#concept-element-disabled
// https://html.spec.whatwg.org/#concept-fieldset-disabled
bool HTMLFieldSetElement::isActuallyDisabled() const
{
    return HTMLFormControlElement::isDisabledFormControl();
}

static void updateFromControlElementsAncestorDisabledStateUnder(HTMLElement& startNode, bool isDisabled)
{
    auto range = inclusiveDescendantsOfType<Element>(startNode);
    auto it = range.begin();

    // This preserves status-quo established before https://github.com/WebKit/WebKit/pull/4988:
    // setDisabledByAncestorFieldset() may modify shadow DOM of <input> / <textarea> elements, but we don't care.
    it.dropAssertions();

    while (it != range.end()) {
        if (auto* listedElement = it->asValidatedFormListedElement())
            listedElement->setDisabledByAncestorFieldset(isDisabled);

        // Don't call setDisabledByAncestorFieldset() on form controls inside disabled fieldsets.
        if (is<HTMLFieldSetElement>(*it) && it->hasAttributeWithoutSynchronization(disabledAttr))
            it.traverseNextSkippingChildren();
        else
            it.traverseNext();
    }
}

void HTMLFieldSetElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    bool oldHasDisabledAttribute = hasDisabledAttribute();

    HTMLFormControlElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);

    if (hasDisabledAttribute() != oldHasDisabledAttribute) {
        if (hasDisabledAttribute())
            document().addDisabledFieldsetElement();
        else
            document().removeDisabledFieldsetElement();
    }
}

void HTMLFieldSetElement::disabledStateChanged()
{
    // This element must be updated before the style of nodes in its subtree gets recalculated.
    HTMLFormControlElement::disabledStateChanged();

    if (disabledByAncestorFieldset())
        return;

    bool thisFieldsetIsDisabled = hasAttributeWithoutSynchronization(disabledAttr);
    bool hasSeenFirstLegendElement = false;
    for (RefPtr<HTMLElement> control = Traversal<HTMLElement>::firstChild(*this); control; control = Traversal<HTMLElement>::nextSibling(*control)) {
        if (!hasSeenFirstLegendElement && is<HTMLLegendElement>(*control)) {
            hasSeenFirstLegendElement = true;
            updateFromControlElementsAncestorDisabledStateUnder(*control, false /* isDisabled */);
            continue;
        }
        updateFromControlElementsAncestorDisabledStateUnder(*control, thisFieldsetIsDisabled);
    }
}

void HTMLFieldSetElement::childrenChanged(const ChildChange& change)
{
    HTMLFormControlElement::childrenChanged(change);
    if (!hasAttributeWithoutSynchronization(disabledAttr))
        return;

    RefPtr<HTMLLegendElement> legend = Traversal<HTMLLegendElement>::firstChild(*this);
    if (!legend)
        return;

    // We only care about the first legend element (in which form controls are not disabled by this element) changing here.
    updateFromControlElementsAncestorDisabledStateUnder(*legend, false /* isDisabled */);
    while ((legend = Traversal<HTMLLegendElement>::nextSibling(*legend)))
        updateFromControlElementsAncestorDisabledStateUnder(*legend, true);
}

void HTMLFieldSetElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    ASSERT_WITH_SECURITY_IMPLICATION(&document() == &newDocument);
    HTMLFormControlElement::didMoveToNewDocument(oldDocument, newDocument);
    if (hasDisabledAttribute()) {
        oldDocument.removeDisabledFieldsetElement();
        newDocument.addDisabledFieldsetElement();
    }
}

bool HTMLFieldSetElement::matchesValidPseudoClass() const
{
    return m_invalidDescendants.isEmptyIgnoringNullReferences();
}

bool HTMLFieldSetElement::matchesInvalidPseudoClass() const
{
    return !m_invalidDescendants.isEmptyIgnoringNullReferences();
}

bool HTMLFieldSetElement::supportsFocus() const
{
    return HTMLElement::supportsFocus() && !isDisabledFormControl();
}

const AtomString& HTMLFieldSetElement::formControlType() const
{
    static MainThreadNeverDestroyed<const AtomString> fieldset("fieldset"_s);
    return fieldset;
}

RenderPtr<RenderElement> HTMLFieldSetElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    // Fieldsets should make a block flow if display: inline or table types are set.
    return RenderElement::createFor(*this, WTFMove(style), { RenderElement::ConstructBlockLevelRendererFor::Inline, RenderElement::ConstructBlockLevelRendererFor::TableOrTablePart });
}

HTMLLegendElement* HTMLFieldSetElement::legend() const
{
    return const_cast<HTMLLegendElement*>(childrenOfType<HTMLLegendElement>(*this).first());
}

Ref<HTMLCollection> HTMLFieldSetElement::elements()
{
    return ensureRareData().ensureNodeLists().addCachedCollection<GenericCachedHTMLCollection<CollectionTypeTraits<CollectionType::FieldSetElements>::traversalType>>(*this, CollectionType::FieldSetElements);
}

void HTMLFieldSetElement::addInvalidDescendant(const HTMLElement& invalidFormControlElement)
{
    ASSERT_WITH_MESSAGE(!is<HTMLFieldSetElement>(invalidFormControlElement), "FieldSet are never candidates for constraint validation.");
    ASSERT(static_cast<const Element&>(invalidFormControlElement).matchesInvalidPseudoClass());
    ASSERT_WITH_MESSAGE(!m_invalidDescendants.contains(invalidFormControlElement), "Updating the fieldset on validity change is not an efficient operation, it should only be done when necessary.");

    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (m_invalidDescendants.isEmptyIgnoringNullReferences())
        emplace(styleInvalidation, *this, { { CSSSelector::PseudoClass::Valid, false }, { CSSSelector::PseudoClass::Invalid, true } });

    m_invalidDescendants.add(invalidFormControlElement);
}

void HTMLFieldSetElement::removeInvalidDescendant(const HTMLElement& formControlElement)
{
    ASSERT_WITH_MESSAGE(!is<HTMLFieldSetElement>(formControlElement), "FieldSet are never candidates for constraint validation.");
    ASSERT_WITH_MESSAGE(m_invalidDescendants.contains(formControlElement), "Updating the fieldset on validity change is not an efficient operation, it should only be done when necessary.");

    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (m_invalidDescendants.computeSize() == 1)
        emplace(styleInvalidation, *this, { { CSSSelector::PseudoClass::Valid, true }, { CSSSelector::PseudoClass::Invalid, false } });

    m_invalidDescendants.remove(formControlElement);
}

} // namespace
