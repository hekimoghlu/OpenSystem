/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class TreeWalker final : public ScriptWrappable, public RefCounted<TreeWalker>, public NodeIteratorBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(TreeWalker, WEBCORE_EXPORT);
public:
    static Ref<TreeWalker> create(Node& rootNode, unsigned long whatToShow, RefPtr<NodeFilter>&& filter)
    {
        return adoptRef(*new TreeWalker(rootNode, whatToShow, WTFMove(filter)));
    }                            

    Node& currentNode() { return m_current.get(); }
    const Node& currentNode() const { return m_current.get(); }

    WEBCORE_EXPORT void setCurrentNode(Node&);

    WEBCORE_EXPORT ExceptionOr<Node*> parentNode();
    WEBCORE_EXPORT ExceptionOr<Node*> firstChild();
    WEBCORE_EXPORT ExceptionOr<Node*> lastChild();
    WEBCORE_EXPORT ExceptionOr<Node*> previousSibling();
    WEBCORE_EXPORT ExceptionOr<Node*> nextSibling();
    WEBCORE_EXPORT ExceptionOr<Node*> previousNode();
    WEBCORE_EXPORT ExceptionOr<Node*> nextNode();

private:
    TreeWalker(Node&, unsigned long whatToShow, RefPtr<NodeFilter>&&);

    enum class SiblingTraversalType { Previous, Next };
    template<SiblingTraversalType> ExceptionOr<Node*> traverseSiblings();
    
    Node* setCurrent(Ref<Node>&&);

    Ref<Node> m_current;
};

} // namespace WebCore
