/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class AbstractSlotVisitor;

}

namespace WebCore {

class CharacterData;
class ContainerNode;
class Element;
class Node;
class NodeList;
class QualifiedName;

class MutationRecord : public RefCounted<MutationRecord> {
public:
    static Ref<MutationRecord> createChildList(ContainerNode& target, Ref<NodeList>&& added, Ref<NodeList>&& removed, RefPtr<Node>&& previousSibling, RefPtr<Node>&& nextSibling);
    static Ref<MutationRecord> createAttributes(Element& target, const QualifiedName&, const AtomString& oldValue);
    static Ref<MutationRecord> createCharacterData(CharacterData& target, const String& oldValue);

    static Ref<MutationRecord> createWithNullOldValue(MutationRecord&);

    virtual ~MutationRecord();

    virtual const AtomString& type() = 0;
    virtual Node* target() = 0;

    virtual NodeList* addedNodes() = 0;
    virtual NodeList* removedNodes() = 0;
    virtual Node* previousSibling() { return 0; }
    virtual Node* nextSibling() { return 0; }

    virtual const AtomString& attributeName() { return nullAtom(); }
    virtual const AtomString& attributeNamespace() { return nullAtom(); }

    virtual String oldValue() { return String(); }

    virtual void visitNodesConcurrently(JSC::AbstractSlotVisitor&) const = 0;
};

} // namespace WebCore
