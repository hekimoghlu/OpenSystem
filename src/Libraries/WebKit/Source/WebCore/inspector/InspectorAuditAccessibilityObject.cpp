/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#include "InspectorAuditAccessibilityObject.h"

#include "AXCoreObject.h"
#include "AXObjectCache.h"
#include "AccessibilityNodeObject.h"
#include "ContainerNode.h"
#include "Document.h"
#include "HTMLNames.h"
#include "SpaceSplitString.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace Inspector;

#define ERROR_IF_NO_ACTIVE_AUDIT() \
    if (!m_auditAgent.hasActiveAudit()) \
        return Exception { ExceptionCode::NotAllowedError, "Cannot be called outside of a Web Inspector Audit"_s };

InspectorAuditAccessibilityObject::InspectorAuditAccessibilityObject(InspectorAuditAgent& auditAgent)
    : m_auditAgent(auditAgent)
{
}

static AccessibilityObject* accessibilityObjectForNode(Node& node)
{
    if (!AXObjectCache::accessibilityEnabled())
        AXObjectCache::enableAccessibility();

    if (AXObjectCache* axObjectCache = node.document().axObjectCache())
        return axObjectCache->getOrCreate(node);

    return nullptr;
}

ExceptionOr<Vector<Ref<Node>>> InspectorAuditAccessibilityObject::getElementsByComputedRole(Document& document, const String& role, Node* container)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    Vector<Ref<Node>> nodes;

    auto* containerNode = dynamicDowncast<ContainerNode>(container);
    for (Element& element : descendantsOfType<Element>(containerNode ? *containerNode : document)) {
        if (auto* axObject = accessibilityObjectForNode(element)) {
            if (axObject->computedRoleString() == role)
                nodes.append(element);
        }
    }

    return nodes;
}

ExceptionOr<RefPtr<Node>> InspectorAuditAccessibilityObject::getActiveDescendant(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    if (auto* axObject = accessibilityObjectForNode(node)) {
        if (AXCoreObject* activeDescendant = axObject->activeDescendant())
            return activeDescendant->node();
    }

    return nullptr;
}

static void addChildren(AXCoreObject& parentObject, Vector<Ref<Node>>& childNodes)
{
    for (const auto& childObject : parentObject.unignoredChildren()) {
        if (RefPtr childNode = childObject->node())
            childNodes.append(childNode.releaseNonNull());
        else
            addChildren(childObject.get(), childNodes);
    }
}

ExceptionOr<std::optional<Vector<Ref<Node>>>> InspectorAuditAccessibilityObject::getChildNodes(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<Vector<Ref<Node>>> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        Vector<Ref<Node>> childNodes;
        addChildren(*axObject, childNodes);
        result = WTFMove(childNodes);
    }

    return result;
}

ExceptionOr<std::optional<InspectorAuditAccessibilityObject::ComputedProperties>> InspectorAuditAccessibilityObject::getComputedProperties(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<InspectorAuditAccessibilityObject::ComputedProperties> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        ComputedProperties computedProperties;

        AXCoreObject* current = axObject;
        while (current && (!computedProperties.busy || !computedProperties.busy.value())) {
            computedProperties.busy = current->isBusy();
            current = current->parentObject();
        }

        if (axObject->supportsChecked()) {
            AccessibilityButtonState checkValue = axObject->checkboxOrRadioValue();
            if (checkValue == AccessibilityButtonState::On)
                computedProperties.checked = "true"_s;
            else if (checkValue == AccessibilityButtonState::Mixed)
                computedProperties.checked = "mixed"_s;
            else if (axObject->isChecked())
                computedProperties.checked = "true"_s;
            else
                computedProperties.checked = "false"_s;
        }

        switch (axObject->currentState()) {
        case AccessibilityCurrentState::False:
            computedProperties.currentState = "false"_s;
            break;
        case AccessibilityCurrentState::True:
            computedProperties.currentState = "true"_s;
            break;
        case AccessibilityCurrentState::Page:
            computedProperties.currentState = "page"_s;
            break;
        case AccessibilityCurrentState::Step:
            computedProperties.currentState = "step"_s;
            break;
        case AccessibilityCurrentState::Location:
            computedProperties.currentState = "location"_s;
            break;
        case AccessibilityCurrentState::Date:
            computedProperties.currentState = "date"_s;
            break;
        case AccessibilityCurrentState::Time:
            computedProperties.currentState = "time"_s;
            break;
        }

        computedProperties.disabled = !axObject->isEnabled();

        if (axObject->supportsExpanded())
            computedProperties.expanded = axObject->isExpanded();

        if (is<Element>(node) && axObject->canSetFocusAttribute())
            computedProperties.focused = axObject->isFocused();

        computedProperties.headingLevel = axObject->headingLevel();
        computedProperties.hidden = axObject->isHidden();
        computedProperties.hierarchicalLevel = axObject->hierarchicalLevel();
        computedProperties.ignored = axObject->isIgnored();
        computedProperties.ignoredByDefault = axObject->isIgnoredByDefault();

        String invalidValue = axObject->invalidStatus();
        if (invalidValue == "false"_s)
            computedProperties.invalidStatus = "false"_s;
        else if (invalidValue == "grammar"_s)
            computedProperties.invalidStatus = "grammar"_s;
        else if (invalidValue == "spelling"_s)
            computedProperties.invalidStatus = "spelling"_s;
        else
            computedProperties.invalidStatus = "true"_s;

        computedProperties.isPopUpButton = axObject->isPopUpButton() || axObject->selfOrAncestorLinkHasPopup();
        computedProperties.label = axObject->computedLabel();

        if (axObject->supportsLiveRegion()) {
            computedProperties.liveRegionAtomic = axObject->liveRegionAtomic();

            String ariaRelevantAttrValue = axObject->liveRegionRelevant();
            if (!ariaRelevantAttrValue.isEmpty()) {
                Vector<String> liveRegionRelevant;
                AtomString ariaRelevantAdditions = "additions"_s;
                AtomString ariaRelevantRemovals = "removals"_s;
                AtomString ariaRelevantText = "text"_s;

                const auto& values = SpaceSplitString(AtomString { ariaRelevantAttrValue }, SpaceSplitString::ShouldFoldCase::Yes);
                if (values.contains("all"_s)) {
                    liveRegionRelevant.append(ariaRelevantAdditions);
                    liveRegionRelevant.append(ariaRelevantRemovals);
                    liveRegionRelevant.append(ariaRelevantText);
                } else {
                    if (values.contains(ariaRelevantAdditions))
                        liveRegionRelevant.append(ariaRelevantAdditions);
                    if (values.contains(ariaRelevantRemovals))
                        liveRegionRelevant.append(ariaRelevantRemovals);
                    if (values.contains(ariaRelevantText))
                        liveRegionRelevant.append(ariaRelevantText);
                }
                computedProperties.liveRegionRelevant = liveRegionRelevant;
            }

            computedProperties.liveRegionStatus = axObject->liveRegionStatus();
        }

        computedProperties.pressed = axObject->pressedIsPresent() && axObject->isPressed();

        if (axObject->isTextControl())
            computedProperties.readonly = !axObject->canSetValueAttribute();

        if (axObject->supportsRequiredAttribute())
            computedProperties.required = axObject->isRequired();

        computedProperties.role = axObject->computedRoleString();
        computedProperties.selected = axObject->isSelected();

        result = computedProperties;
    }

    return result;
}

ExceptionOr<std::optional<Vector<Ref<Node>>>> InspectorAuditAccessibilityObject::getControlledNodes(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<Vector<Ref<Node>>> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        auto controlledElements = axObject->elementsFromAttribute(HTMLNames::aria_controlsAttr);
        result = WTF::map(WTFMove(controlledElements), [](auto&& element) -> Ref<Node> {
            return WTFMove(element);
        });
    }

    return result;
}

ExceptionOr<std::optional<Vector<Ref<Node>>>> InspectorAuditAccessibilityObject::getFlowedNodes(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<Vector<Ref<Node>>> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        auto flowedElements = axObject->elementsFromAttribute(HTMLNames::aria_flowtoAttr);
        result = WTF::map(WTFMove(flowedElements), [](auto&& element) -> Ref<Node> {
            return WTFMove(element);
        });
    }

    return result;
}

ExceptionOr<RefPtr<Node>> InspectorAuditAccessibilityObject::getMouseEventNode(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    if (auto* axObject = accessibilityObjectForNode(node)) {
        if (auto* clickableObject = axObject->clickableSelfOrAncestor(ClickHandlerFilter::IncludeBody))
            return clickableObject->node();
    }

    return nullptr;
}

ExceptionOr<std::optional<Vector<Ref<Node>>>> InspectorAuditAccessibilityObject::getOwnedNodes(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<Vector<Ref<Node>>> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        if (axObject->supportsARIAOwns()) {
            auto ownedElements = axObject->elementsFromAttribute(HTMLNames::aria_ownsAttr);
            result = WTF::map(WTFMove(ownedElements), [](auto&& element) -> Ref<Node> {
                return WTFMove(element);
            });
        }
    }

    return result;
}

ExceptionOr<RefPtr<Node>> InspectorAuditAccessibilityObject::getParentNode(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    if (auto* axObject = accessibilityObjectForNode(node)) {
        if (AXCoreObject* parentObject = axObject->parentObjectUnignored())
            return parentObject->node();
    }

    return nullptr;
}

ExceptionOr<std::optional<Vector<Ref<Node>>>> InspectorAuditAccessibilityObject::getSelectedChildNodes(Node& node)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    std::optional<Vector<Ref<Node>>> result;

    if (auto* axObject = accessibilityObjectForNode(node)) {
        Vector<Ref<Node>> selectedChildNodes;

        auto selectedChildren = axObject->selectedChildren();
        for (auto& selectedChildObject : selectedChildren) {
            if (RefPtr selectedChildNode = selectedChildObject->node())
                selectedChildNodes.append(selectedChildNode.releaseNonNull());
        }

        result = WTFMove(selectedChildNodes);
    }

    return result;
}

} // namespace WebCore
