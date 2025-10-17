/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#include "NodeFilter.h"
#include "ScriptWrappable.h"
#include "Traversal.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class NodeIterator final : public ScriptWrappable, public RefCountedAndCanMakeWeakPtr<NodeIterator>, public NodeIteratorBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(NodeIterator, WEBCORE_EXPORT);
public:
    static Ref<NodeIterator> create(Node&, unsigned whatToShow, RefPtr<NodeFilter>&&);
    WEBCORE_EXPORT ~NodeIterator();

    WEBCORE_EXPORT ExceptionOr<RefPtr<Node>> nextNode();
    WEBCORE_EXPORT ExceptionOr<RefPtr<Node>> previousNode();
    void detach() { } // This is now a no-op as per the DOM specification.

    Node* referenceNode() const { return m_referenceNode.node.get(); }
    bool pointerBeforeReferenceNode() const { return m_referenceNode.isPointerBeforeNode; }

    // This function is called before any node is removed from the document tree.
    void nodeWillBeRemoved(Node&);

private:
    NodeIterator(Node&, unsigned whatToShow, RefPtr<NodeFilter>&&);

    struct NodePointer {
        RefPtr<Node> node;
        bool isPointerBeforeNode { true };

        NodePointer() = default;
        NodePointer(Node&, bool);
        RefPtr<Node> protectedNode() const { return node; }

        void clear();
        bool moveToNext(Node& root);
        bool moveToPrevious(Node& root);
    };

    void updateForNodeRemoval(Node& nodeToBeRemoved, NodePointer&) const;

    NodePointer m_referenceNode;
    NodePointer m_candidateNode;
};

} // namespace WebCore
