/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

namespace WebCore {

class ContainerNode;
class DOMEditor;
class Document;
class Node;
class WeakPtrImplWithEventTargetData;

class DOMPatchSupport final {
public:
    DOMPatchSupport(DOMEditor&, Document&);

    void patchDocument(const String& markup);
    ExceptionOr<Node*> patchNode(Node&, const String& markup);

private:
    struct Digest;

    using ResultMap = Vector<std::pair<Digest*, size_t>>;
    using UnusedNodesMap = HashMap<String, Digest*>;

    ExceptionOr<void> innerPatchNode(Digest& oldNode, Digest& newNode);
    std::pair<ResultMap, ResultMap> diff(const Vector<std::unique_ptr<Digest>>& oldChildren, const Vector<std::unique_ptr<Digest>>& newChildren);
    ExceptionOr<void> innerPatchChildren(ContainerNode&, const Vector<std::unique_ptr<Digest>>& oldChildren, const Vector<std::unique_ptr<Digest>>& newChildren);
    std::unique_ptr<Digest> createDigest(Node&, UnusedNodesMap*);
    ExceptionOr<void> insertBeforeAndMarkAsUsed(ContainerNode&, Digest&, Node* anchor);
    ExceptionOr<void> removeChildAndMoveToNew(Digest&);
    void markNodeAsUsed(Digest&);

#ifdef DEBUG_DOM_PATCH_SUPPORT
    void dumpMap(const ResultMap&, const String& name);
#endif

    DOMEditor& m_domEditor;
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;

    UnusedNodesMap m_unusedNodesMap;
};

} // namespace WebCore
