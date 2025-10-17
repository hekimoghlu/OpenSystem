/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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

#include "Element.h"
#include "NodeList.h"
#include <wtf/Vector.h>

namespace WebCore {

class WEBCORE_EXPORT StaticNodeList final : public NodeList {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(StaticNodeList, WEBCORE_EXPORT);
public:
    static Ref<StaticNodeList> create(Vector<Ref<Node>>&& nodes = { })
    {
        return adoptRef(*new StaticNodeList(WTFMove(nodes)));
    }

    unsigned length() const override;
    Node* item(unsigned index) const override;

private:
    StaticNodeList(Vector<Ref<Node>>&& nodes)
        : m_nodes(WTFMove(nodes))
    { }

    Vector<Ref<Node>> m_nodes;
};

class StaticWrapperNodeList final : public NodeList {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StaticWrapperNodeList);
public:
    static Ref<StaticWrapperNodeList> create(NodeList& nodeList)
    {
        return adoptRef(*new StaticWrapperNodeList(nodeList));
    }

    unsigned length() const override;
    Node* item(unsigned index) const override;

private:
    StaticWrapperNodeList(NodeList& nodeList)
        : m_nodeList(nodeList)
    { }

    Ref<NodeList>  m_nodeList;
};

class StaticElementList final : public NodeList {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StaticElementList);
public:
    static Ref<StaticElementList> create(Vector<Ref<Element>>&& elements = { })
    {
        return adoptRef(*new StaticElementList(WTFMove(elements)));
    }

    unsigned length() const override;
    Element* item(unsigned index) const override;

private:
    StaticElementList(Vector<Ref<Element>>&& elements)
        : m_elements(WTFMove(elements))
    { }

    Vector<Ref<Element>> m_elements;
};

} // namespace WebCore
