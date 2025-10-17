/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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

#include "HTMLElement.h"

namespace WebCore {

class HTMLSlotElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLSlotElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLSlotElement);
public:
    using ElementOrText = std::variant<RefPtr<Element>, RefPtr<Text>>;

    static Ref<HTMLSlotElement> create(const QualifiedName&, Document&);

    const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>* assignedNodes() const;
    struct AssignedNodesOptions {
        bool flatten;
    };
    Vector<Ref<Node>> assignedNodes(const AssignedNodesOptions&) const;
    Vector<Ref<Element>> assignedElements(const AssignedNodesOptions&) const;

    void assign(FixedVector<ElementOrText>&&);
    const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& manuallyAssignedNodes() const { return m_manuallyAssignedNodes; }
    void removeManuallyAssignedNode(Node&);

    void enqueueSlotChangeEvent();
    void didRemoveFromSignalSlotList() { m_inSignalSlotList = false; }

    void dispatchSlotChangeEvent();

    bool isInInsertedIntoAncestor() const { return m_isInInsertedIntoAncestor; }

private:
    HTMLSlotElement(const QualifiedName&, Document&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void childrenChanged(const ChildChange&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void didFinishInsertingNode() final;

    bool m_inSignalSlotList { false };
    bool m_isInInsertedIntoAncestor { false };
    Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>> m_manuallyAssignedNodes;
};

} // namespace WebCore
