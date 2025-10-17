/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include "HTMLSlotElement.h"

#include "ElementInlines.h"
#include "Event.h"
#include "EventNames.h"
#include "HTMLNames.h"
#include "MutationObserver.h"
#include "ShadowRoot.h"
#include "SlotAssignment.h"
#include "Text.h"
#include <wtf/SetForScope.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLSlotElement);

using namespace HTMLNames;

Ref<HTMLSlotElement> HTMLSlotElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLSlotElement(tagName, document));
}

HTMLSlotElement::HTMLSlotElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(slotTag));
}

HTMLSlotElement::InsertedIntoAncestorResult HTMLSlotElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    SetForScope isInInsertedIntoAncestor { m_isInInsertedIntoAncestor, true };

    auto insertionResult = HTMLElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    ASSERT_UNUSED(insertionResult, insertionResult == InsertedIntoAncestorResult::Done);

    if (insertionType.treeScopeChanged && isInShadowTree()) {
        if (auto* shadowRoot = containingShadowRoot())
            shadowRoot->addSlotElementByName(attributeWithoutSynchronization(nameAttr), *this);
    }

    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void HTMLSlotElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    if (removalType.treeScopeChanged && oldParentOfRemovedTree.isInShadowTree()) {
        auto* oldShadowRoot = oldParentOfRemovedTree.containingShadowRoot();
        ASSERT(oldShadowRoot);
        oldShadowRoot->removeSlotElementByName(attributeWithoutSynchronization(nameAttr), *this, oldParentOfRemovedTree);
    }

    HTMLElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
}

void HTMLSlotElement::childrenChanged(const ChildChange& childChange)
{
    HTMLElement::childrenChanged(childChange);

    if (isInShadowTree()) {
        if (auto* shadowRoot = containingShadowRoot())
            shadowRoot->slotFallbackDidChange(*this);
    }
}

void HTMLSlotElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason reason)
{
    HTMLElement::attributeChanged(name, oldValue, newValue, reason);

    if (isInShadowTree() && name == nameAttr) {
        if (RefPtr shadowRoot = containingShadowRoot())
            shadowRoot->renameSlotElement(*this, oldValue, newValue);
    }
}

void HTMLSlotElement::didFinishInsertingNode()
{
    HTMLElement::didFinishInsertingNode();
    if (selfOrPrecedingNodesAffectDirAuto())
        updateEffectiveTextDirection();
}

const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>* HTMLSlotElement::assignedNodes() const
{
    RefPtr shadowRoot = containingShadowRoot();
    if (!shadowRoot)
        return nullptr;

    return shadowRoot->assignedNodesForSlot(*this);
}

static void flattenAssignedNodes(Vector<Ref<Node>>& nodes, const HTMLSlotElement& slot)
{
    if (!slot.containingShadowRoot())
        return;

    auto* assignedNodes = slot.assignedNodes();
    if (!assignedNodes) {
        for (RefPtr<Node> child = slot.firstChild(); child; child = child->nextSibling()) {
            if (auto* slot = dynamicDowncast<HTMLSlotElement>(*child))
                flattenAssignedNodes(nodes, *slot);
            else if (is<Text>(*child) || is<Element>(*child))
                nodes.append(*child);
        }
        return;
    }
    for (auto& weakNode : *assignedNodes) {
        if (UNLIKELY(!weakNode)) {
            ASSERT_NOT_REACHED();
            continue;
        }
        if (RefPtr slot = dynamicDowncast<HTMLSlotElement>(*weakNode); slot && slot->containingShadowRoot())
            flattenAssignedNodes(nodes, *slot);
        else
            nodes.append(Ref { *weakNode });
    }
}

Vector<Ref<Node>> HTMLSlotElement::assignedNodes(const AssignedNodesOptions& options) const
{
    if (options.flatten) {
        if (!isInShadowTree())
            return { };
        Vector<Ref<Node>> nodes;
        flattenAssignedNodes(nodes, *this);
        return nodes;
    }

    if (auto* nodes = assignedNodes()) {
        return compactMap(*nodes, [](auto& nodeWeakPtr) -> RefPtr<Node> {
            return nodeWeakPtr.get();
        });
    }

    return { };
}

Vector<Ref<Element>> HTMLSlotElement::assignedElements(const AssignedNodesOptions& options) const
{
    return compactMap(assignedNodes(options), [](Ref<Node>&& node) -> RefPtr<Element> {
        return dynamicDowncast<Element>(WTFMove(node));
    });
}

void HTMLSlotElement::assign(FixedVector<ElementOrText>&& nodes)
{
    RefPtr shadowRoot = containingShadowRoot();
    RefPtr host = shadowRoot ? shadowRoot->host() : nullptr;
    for (auto& node : m_manuallyAssignedNodes) {
        if (RefPtr protectedNode = node.get())
            protectedNode->setManuallyAssignedSlot(nullptr);
    }

    auto previous = std::exchange(m_manuallyAssignedNodes, { });
    UncheckedKeyHashSet<RefPtr<Node>> seenNodes;
    m_manuallyAssignedNodes = WTF::compactMap(nodes, [&seenNodes](ElementOrText& node) -> std::optional<WeakPtr<Node, WeakPtrImplWithEventTargetData>> {
        auto mapper = [&seenNodes]<typename T>(RefPtr<T>& node) -> std::optional<WeakPtr<Node, WeakPtrImplWithEventTargetData>> {
            if (seenNodes.contains(node))
                return std::nullopt;
            seenNodes.add(node);
            return WeakPtr { node };
        };

        return WTF::switchOn(node,
            [&mapper](RefPtr<Element>& node) { return mapper(node); },
            [&mapper](RefPtr<Text>& node) { return mapper(node); }
        );
    });

    if (RefPtr shadowRoot = containingShadowRoot(); shadowRoot && shadowRoot->slotAssignmentMode() == SlotAssignmentMode::Manual)
        shadowRoot->slotManualAssignmentDidChange(*this, previous, m_manuallyAssignedNodes);
    else {
        for (auto& node : m_manuallyAssignedNodes) {
            if (auto previousSlot = node->manuallyAssignedSlot()) {
                previousSlot->removeManuallyAssignedNode(*node);
                if (RefPtr shadowRootOfPreviousSlot = previousSlot->containingShadowRoot(); shadowRootOfPreviousSlot && node->parentNode() == shadowRootOfPreviousSlot->host())
                    shadowRootOfPreviousSlot->didRemoveManuallyAssignedNode(*previousSlot, *node);
            }
            node->setManuallyAssignedSlot(this);
        }
    }
}

void HTMLSlotElement::removeManuallyAssignedNode(Node& node)
{
    m_manuallyAssignedNodes.removeFirst(&node);
}

void HTMLSlotElement::enqueueSlotChangeEvent()
{
    // https://dom.spec.whatwg.org/#signal-a-slot-change
    if (m_inSignalSlotList)
        return;
    m_inSignalSlotList = true;
    MutationObserver::enqueueSlotChangeEvent(*this);
}

void HTMLSlotElement::dispatchSlotChangeEvent()
{
    m_inSignalSlotList = false;

    Ref<Event> event = Event::create(eventNames().slotchangeEvent, Event::CanBubble::Yes, Event::IsCancelable::No);
    event->setTarget(Ref { *this });
    dispatchEvent(event);
}

} // namespace WebCore
