/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#include "ExceptionOr.h"
#include <JavaScriptCore/InspectorAuditAgent.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class Document;
class Node;

class InspectorAuditAccessibilityObject : public RefCounted<InspectorAuditAccessibilityObject> {
public:
    static Ref<InspectorAuditAccessibilityObject> create(Inspector::InspectorAuditAgent& auditAgent)
    {
        return adoptRef(*new InspectorAuditAccessibilityObject(auditAgent));
    }

    ExceptionOr<Vector<Ref<Node>>> getElementsByComputedRole(Document&, const String& role, Node* container);

    struct ComputedProperties {
        std::optional<bool> busy;
        String checked;
        String currentState;
        std::optional<bool> disabled;
        std::optional<bool> expanded;
        std::optional<bool> focused;
        std::optional<int> headingLevel;
        std::optional<bool> hidden;
        std::optional<int> hierarchicalLevel;
        std::optional<bool> ignored;
        std::optional<bool> ignoredByDefault;
        String invalidStatus;
        std::optional<bool> isPopUpButton;
        String label;
        std::optional<bool> liveRegionAtomic;
        std::optional<Vector<String>> liveRegionRelevant;
        String liveRegionStatus;
        std::optional<bool> pressed;
        std::optional<bool> readonly;
        std::optional<bool> required;
        String role;
        std::optional<bool> selected;
    };

    ExceptionOr<RefPtr<Node>> getActiveDescendant(Node&);
    ExceptionOr<std::optional<Vector<Ref<Node>>>> getChildNodes(Node&);
    ExceptionOr<std::optional<ComputedProperties>> getComputedProperties(Node&);
    ExceptionOr<std::optional<Vector<Ref<Node>>>> getControlledNodes(Node&);
    ExceptionOr<std::optional<Vector<Ref<Node>>>> getFlowedNodes(Node&);
    ExceptionOr<RefPtr<Node>> getMouseEventNode(Node&);
    ExceptionOr<std::optional<Vector<Ref<Node>>>> getOwnedNodes(Node&);
    ExceptionOr<RefPtr<Node>> getParentNode(Node&);
    ExceptionOr<std::optional<Vector<Ref<Node>>>> getSelectedChildNodes(Node&);

private:
    explicit InspectorAuditAccessibilityObject(Inspector::InspectorAuditAgent&);

    Inspector::InspectorAuditAgent& m_auditAgent;
};

} // namespace WebCore
