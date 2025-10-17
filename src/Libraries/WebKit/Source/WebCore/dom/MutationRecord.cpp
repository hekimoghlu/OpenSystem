/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#include "MutationRecord.h"

#include "CharacterData.h"
#include "JSNode.h"
#include "StaticNodeList.h"
#include "WebCoreOpaqueRootInlines.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

namespace {

static void visitNodeList(JSC::AbstractSlotVisitor& visitor, NodeList& nodeList)
{
    ASSERT(!nodeList.isLiveNodeList());
    unsigned length = nodeList.length();
    for (unsigned i = 0; i < length; ++i)
        addWebCoreOpaqueRoot(visitor, nodeList.item(i));
}

class ChildListRecord final : public MutationRecord {
public:
    ChildListRecord(ContainerNode& target, Ref<NodeList>&& added, Ref<NodeList>&& removed, RefPtr<Node>&& previousSibling, RefPtr<Node>&& nextSibling)
        : m_target(target)
        , m_addedNodes(WTFMove(added))
        , m_removedNodes(WTFMove(removed))
        , m_previousSibling(WTFMove(previousSibling))
        , m_nextSibling(WTFMove(nextSibling))
    {
    }

private:
    const AtomString& type() override;
    Node* target() override { return m_target.ptr(); }
    NodeList* addedNodes() override { return m_addedNodes.get(); }
    NodeList* removedNodes() override { return m_removedNodes.get(); }
    Node* previousSibling() override { return m_previousSibling.get(); }
    Node* nextSibling() override { return m_nextSibling.get(); }

    void visitNodesConcurrently(JSC::AbstractSlotVisitor& visitor) const final
    {
        addWebCoreOpaqueRoot(visitor, m_target.get());
        if (m_addedNodes)
            visitNodeList(visitor, *m_addedNodes);
        if (m_removedNodes)
            visitNodeList(visitor, *m_removedNodes);
    }
    
    Ref<ContainerNode> m_target;
    RefPtr<NodeList> m_addedNodes;
    RefPtr<NodeList> m_removedNodes;
    RefPtr<Node> m_previousSibling;
    RefPtr<Node> m_nextSibling;
};

class RecordWithEmptyNodeLists : public MutationRecord {
public:
    RecordWithEmptyNodeLists(Node& target, const String& oldValue)
        : m_target(target)
        , m_oldValue(oldValue)
    {
    }

private:
    Node* target() override { return m_target.ptr(); }
    String oldValue() override { return m_oldValue; }
    NodeList* addedNodes() override { return lazilyInitializeEmptyNodeList(m_addedNodes); }
    NodeList* removedNodes() override { return lazilyInitializeEmptyNodeList(m_removedNodes); }

    static NodeList* lazilyInitializeEmptyNodeList(RefPtr<NodeList>& nodeList)
    {
        if (!nodeList)
            nodeList = StaticNodeList::create();
        return nodeList.get();
    }

    void visitNodesConcurrently(JSC::AbstractSlotVisitor& visitor) const final
    {
        addWebCoreOpaqueRoot(visitor, m_target.get());
    }

    Ref<Node> m_target;
    String m_oldValue;
    RefPtr<NodeList> m_addedNodes;
    RefPtr<NodeList> m_removedNodes;
};

class AttributesRecord final : public RecordWithEmptyNodeLists {
public:
    AttributesRecord(Element& target, const QualifiedName& name, const AtomString& oldValue)
        : RecordWithEmptyNodeLists(target, oldValue)
        , m_attributeName(name.localName())
        , m_attributeNamespace(name.namespaceURI())
    {
    }

private:
    const AtomString& type() override;
    const AtomString& attributeName() override { return m_attributeName; }
    const AtomString& attributeNamespace() override { return m_attributeNamespace; }

    AtomString m_attributeName;
    AtomString m_attributeNamespace;
};

class CharacterDataRecord : public RecordWithEmptyNodeLists {
public:
    CharacterDataRecord(CharacterData& target, const String& oldValue)
        : RecordWithEmptyNodeLists(target, oldValue)
    {
    }

private:
    const AtomString& type() override;
};

class MutationRecordWithNullOldValue final : public MutationRecord {
public:
    MutationRecordWithNullOldValue(MutationRecord& record)
        : m_record(record)
    {
    }

private:
    const AtomString& type() override { return m_record->type(); }
    Node* target() override { return m_record->target(); }
    NodeList* addedNodes() override { return m_record->addedNodes(); }
    NodeList* removedNodes() override { return m_record->removedNodes(); }
    Node* previousSibling() override { return m_record->previousSibling(); }
    Node* nextSibling() override { return m_record->nextSibling(); }
    const AtomString& attributeName() override { return m_record->attributeName(); }
    const AtomString& attributeNamespace() override { return m_record->attributeNamespace(); }

    String oldValue() override { return String(); }

    void visitNodesConcurrently(JSC::AbstractSlotVisitor& visitor) const final
    {
        m_record->visitNodesConcurrently(visitor);
    }

    Ref<MutationRecord> m_record;
};

const AtomString& ChildListRecord::type()
{
    static MainThreadNeverDestroyed<const AtomString> childList("childList"_s);
    return childList;
}

const AtomString& AttributesRecord::type()
{
    static MainThreadNeverDestroyed<const AtomString> attributes("attributes"_s);
    return attributes;
}

const AtomString& CharacterDataRecord::type()
{
    static MainThreadNeverDestroyed<const AtomString> characterData("characterData"_s);
    return characterData;
}

} // namespace

Ref<MutationRecord> MutationRecord::createChildList(ContainerNode& target, Ref<NodeList>&& added, Ref<NodeList>&& removed, RefPtr<Node>&& previousSibling, RefPtr<Node>&& nextSibling)
{
    return adoptRef(static_cast<MutationRecord&>(*new ChildListRecord(target, WTFMove(added), WTFMove(removed), WTFMove(previousSibling), WTFMove(nextSibling))));
}

Ref<MutationRecord> MutationRecord::createAttributes(Element& target, const QualifiedName& name, const AtomString& oldValue)
{
    return adoptRef(static_cast<MutationRecord&>(*new AttributesRecord(target, name, oldValue)));
}

Ref<MutationRecord> MutationRecord::createCharacterData(CharacterData& target, const String& oldValue)
{
    return adoptRef(static_cast<MutationRecord&>(*new CharacterDataRecord(target, oldValue)));
}

Ref<MutationRecord> MutationRecord::createWithNullOldValue(MutationRecord& record)
{
    return adoptRef(static_cast<MutationRecord&>(*new MutationRecordWithNullOldValue(record)));
}

MutationRecord::~MutationRecord() = default;

} // namespace WebCore
