/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

#include "CollectionIndexCache.h"
#include "NodeList.h"
#include <wtf/CheckedRef.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ContainerNode;

class EmptyNodeList final : public NodeList, public CanMakeSingleThreadWeakPtr<EmptyNodeList> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(EmptyNodeList);
public:
    static Ref<EmptyNodeList> create(Node& owner)
    {
        return adoptRef(*new EmptyNodeList(owner));
    }
    virtual ~EmptyNodeList();

    Node& ownerNode() { return m_owner; }

private:
    explicit EmptyNodeList(Node& owner) : m_owner(owner) { }

    unsigned length() const override { return 0; }
    Node* item(unsigned) const override { return nullptr; }
    size_t memoryCost() const override { return 0; }

    bool isEmptyNodeList() const override { return true; }

    Ref<Node> m_owner;
};

class ChildNodeList final : public NodeList, public CanMakeSingleThreadWeakPtr<ChildNodeList> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ChildNodeList);
public:
    static Ref<ChildNodeList> create(ContainerNode& parent)
    {
        return adoptRef(*new ChildNodeList(parent));
    }

    virtual ~ChildNodeList();

    ContainerNode& ownerNode() { return m_parent; }

    void invalidateCache();

    // For CollectionIndexCache
    Node* collectionBegin() const;
    Node* collectionLast() const;
    void collectionTraverseForward(Node*&, unsigned count, unsigned& traversedCount) const;
    void collectionTraverseBackward(Node*&, unsigned count) const;
    bool collectionCanTraverseBackward() const { return true; }
    void willValidateIndexCache() const { }

private:
    explicit ChildNodeList(ContainerNode& parent);

    unsigned length() const override;
    Node* item(unsigned index) const override;
    size_t memoryCost() const override
    {
        // memoryCost() may be invoked concurrently from a GC thread, and we need to be careful
        // about what data we access here and how. Accessing m_indexCache is safe because
        // because it doesn't involve any pointer chasing.
        return m_indexCache.memoryCost();
    }

    bool isChildNodeList() const override { return true; }

    Ref<ContainerNode> m_parent;
    mutable CollectionIndexCache<ChildNodeList, Node*> m_indexCache;
};

} // namespace WebCore
