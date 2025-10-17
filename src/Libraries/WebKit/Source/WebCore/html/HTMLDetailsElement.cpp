/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include "HTMLDetailsElement.h"

#include "AXObjectCache.h"
#include "DocumentInlines.h"
#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "HTMLSlotElement.h"
#include "HTMLStyleElement.h"
#include "HTMLSummaryElement.h"
#include "LocalizedStrings.h"
#include "MouseEvent.h"
#include "ShadowRoot.h"
#include "ShouldNotFireMutationEventsScope.h"
#include "SlotAssignment.h"
#include "Text.h"
#include "ToggleEvent.h"
#include "ToggleEventTask.h"
#include "TypedElementDescendantIteratorInlines.h"
#include "UserAgentParts.h"
#include "UserAgentStyle.h"
#include "UserAgentStyleSheets.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLDetailsElement);

using namespace HTMLNames;

static const AtomString& summarySlotName()
{
    static MainThreadNeverDestroyed<const AtomString> summarySlot("summarySlot"_s);
    return summarySlot;
}

class DetailsSlotAssignment final : public NamedSlotAssignment {
private:
    void hostChildElementDidChange(const Element&, ShadowRoot&) override;
    const AtomString& slotNameForHostChild(const Node&) const override;
};

void DetailsSlotAssignment::hostChildElementDidChange(const Element& childElement, ShadowRoot& shadowRoot)
{
    if (is<HTMLSummaryElement>(childElement)) {
        // Don't check whether this is the first summary element
        // since we don't know the answer when this function is called inside Element::removedFrom.
        didChangeSlot(summarySlotName(), shadowRoot);
    } else
        didChangeSlot(NamedSlotAssignment::defaultSlotName(), shadowRoot);
}

const AtomString& DetailsSlotAssignment::slotNameForHostChild(const Node& child) const
{
    auto& details = downcast<HTMLDetailsElement>(*child.parentNode());

    // The first summary child gets assigned to the summary slot.
    if (is<HTMLSummaryElement>(child)) {
        if (&child == childrenOfType<HTMLSummaryElement>(details).first())
            return summarySlotName();
    }
    return NamedSlotAssignment::defaultSlotName();
}

Ref<HTMLDetailsElement> HTMLDetailsElement::create(const QualifiedName& tagName, Document& document)
{
    auto details = adoptRef(*new HTMLDetailsElement(tagName, document));
    details->addShadowRoot(ShadowRoot::create(document, makeUnique<DetailsSlotAssignment>()));
    return details;
}

HTMLDetailsElement::HTMLDetailsElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(detailsTag));
}

HTMLDetailsElement::~HTMLDetailsElement() = default;

void HTMLDetailsElement::didAddUserAgentShadowRoot(ShadowRoot& root)
{
    auto summarySlot = HTMLSlotElement::create(slotTag, document());
    summarySlot->setAttributeWithoutSynchronization(nameAttr, summarySlotName());
    m_summarySlot = summarySlot.get();

    auto defaultSummary = HTMLSummaryElement::create(summaryTag, document());
    defaultSummary->appendChild(Text::create(document(), defaultDetailsSummaryText()));
    m_defaultSummary = defaultSummary.get();

    summarySlot->appendChild(defaultSummary);
    root.appendChild(summarySlot);

    m_defaultSlot = HTMLSlotElement::create(slotTag, document());
    m_defaultSlot->setUserAgentPart(UserAgentParts::detailsContent());
    ASSERT(!hasAttribute(openAttr));
    m_defaultSlot->setInlineStyleProperty(CSSPropertyContentVisibility, CSSValueHidden);
    m_defaultSlot->setInlineStyleProperty(CSSPropertyDisplay, CSSValueBlock);
    root.appendChild(*m_defaultSlot);

    static MainThreadNeverDestroyed<const String> stylesheet(StringImpl::createWithoutCopying(detailsElementShadowUserAgentStyleSheet));
    auto style = HTMLStyleElement::create(HTMLNames::styleTag, document(), false);
    style->setTextContent(String { stylesheet });
    root.appendChild(WTFMove(style));
}

bool HTMLDetailsElement::isActiveSummary(const HTMLSummaryElement& summary) const
{
    if (!m_summarySlot->assignedNodes())
        return &summary == m_defaultSummary;

    if (summary.parentNode() != this)
        return false;

    RefPtr slot = shadowRoot()->findAssignedSlot(summary);
    if (!slot)
        return false;
    return slot == m_summarySlot.get();
}

void HTMLDetailsElement::queueDetailsToggleEventTask(ToggleState oldState, ToggleState newState)
{
    if (!m_toggleEventTask)
        m_toggleEventTask = ToggleEventTask::create(*this);

    m_toggleEventTask->queue(oldState, newState);
}

void HTMLDetailsElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
    if (name == openAttr) {
        if (oldValue != newValue) {
            RefPtr root = shadowRoot();
            ASSERT(root);
            if (!newValue.isNull()) {
                m_defaultSlot->removeInlineStyleProperty(CSSPropertyContentVisibility);
                queueDetailsToggleEventTask(ToggleState::Closed, ToggleState::Open);
                if (!attributeWithoutSynchronization(nameAttr).isEmpty()) {
                    ShouldNotFireMutationEventsScope scope(document());
                    for (auto& otherDetailsElement : otherElementsInNameGroup())
                        otherDetailsElement->removeAttribute(openAttr);
                }
            } else {
                m_defaultSlot->setInlineStyleProperty(CSSPropertyContentVisibility, CSSValueHidden);
                queueDetailsToggleEventTask(ToggleState::Open, ToggleState::Closed);
            }
        }
    } else
        ensureDetailsExclusivityAfterMutation();
}

Node::InsertedIntoAncestorResult HTMLDetailsElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (!insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::Done;
    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void HTMLDetailsElement::didFinishInsertingNode()
{
    ensureDetailsExclusivityAfterMutation();
}

Vector<RefPtr<HTMLDetailsElement>> HTMLDetailsElement::otherElementsInNameGroup()
{
    Vector<RefPtr<HTMLDetailsElement>> otherElementsInNameGroup;
    const auto& detailElementName = attributeWithoutSynchronization(nameAttr);
    for (auto& element : descendantsOfType<HTMLDetailsElement>(rootNode())) {
        if (&element != this && element.attributeWithoutSynchronization(nameAttr) == detailElementName)
            otherElementsInNameGroup.append(&element);
    }
    return otherElementsInNameGroup;
}

void HTMLDetailsElement::ensureDetailsExclusivityAfterMutation()
{
    if (hasAttribute(openAttr) && !attributeWithoutSynchronization(nameAttr).isEmpty()) {
        ShouldNotFireMutationEventsScope scope(document());
        for (auto& otherDetailsElement : otherElementsInNameGroup()) {
            if (otherDetailsElement->hasAttribute(openAttr)) {
                toggleOpen();
                break;
            }
        }
    }
}

} // namespace WebCore
