/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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

#include "SourceAlpha.h"
#include "SourceGraphic.h"
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FilterEffect;
class SVGFilterPrimitiveStandardAttributes;

template<typename NodeType>
class SVGFilterGraph {
public:
    using NodeVector = Vector<Ref<NodeType>>;

    SVGFilterGraph() = default;

    SVGFilterGraph(Ref<NodeType>&& sourceGraphic, Ref<NodeType>&& sourceAlpha)
    {
        m_sourceNodes.add(SourceGraphic::effectName(), WTFMove(sourceGraphic));
        m_sourceNodes.add(SourceAlpha::effectName(), WTFMove(sourceAlpha));

        setNodeInputs(Ref { *this->sourceGraphic() }, NodeVector { });
        setNodeInputs(Ref { *this->sourceAlpha() }, NodeVector { *this->sourceGraphic() });
    }

    NodeType* sourceGraphic() const
    {
        return m_sourceNodes.get(FilterEffect::sourceGraphicName());
    }

    NodeType* sourceAlpha() const
    {
        return m_sourceNodes.get(FilterEffect::sourceAlphaName());
    }

    void addNamedNode(const AtomString& id, Ref<NodeType>&& node)
    {
        if (id.isEmpty()) {
            m_lastNode = WTFMove(node);
            return;
        }

        if (m_sourceNodes.contains(id))
            return;

        m_lastNode = WTFMove(node);
        m_namedNodes.set(id, Ref { *m_lastNode });
    }

    RefPtr<NodeType> getNamedNode(const AtomString& id) const
    {
        if (!id.isEmpty()) {
            if (auto sourceNode = m_sourceNodes.get(id))
                return sourceNode;

            if (auto namedNode = m_namedNodes.get(id))
                return namedNode;
        }

        if (m_lastNode)
            return m_lastNode;

        // Fallback to the 'sourceGraphic' input.
        return sourceGraphic();
    }

    std::optional<NodeVector> getNamedNodes(std::span<const AtomString> names) const
    {
        NodeVector nodes;

        nodes.reserveInitialCapacity(names.size());

        for (auto& name : names) {
            if (auto node = getNamedNode(name))
                nodes.append(node.releaseNonNull());
            else if (!isSourceName(name))
                return std::nullopt;
        }

        return nodes;
    }

    void setNodeInputs(NodeType& node, NodeVector&& inputs)
    {
        m_nodeInputs.set({ node }, WTFMove(inputs));
    }

    NodeVector getNodeInputs(NodeType& node) const
    {
        return m_nodeInputs.get(node);
    }

    NodeVector nodes() const
    {
        return WTF::map(m_nodeInputs, [] (auto& pair) -> Ref<NodeType> {
            return pair.key;
        });
    }

    NodeType* lastNode() const { return m_lastNode.get(); }

    template<typename Callback>
    bool visit(Callback callback)
    {
        if (!lastNode())
            return false;

        Vector<Ref<NodeType>> stack;
        return visit(*lastNode(), stack, 0, callback);
    }

private:
    static bool isSourceName(const AtomString& id)
    {
        return id == SourceGraphic::effectName() || id == SourceAlpha::effectName();
    }

    template<typename Callback>
    bool visit(NodeType& node, Vector<Ref<NodeType>>& stack, unsigned level, Callback callback)
    {
        // A cycle is detected.
        if (stack.containsIf([&](auto& item) { return item.ptr() == &node; }))
            return false;

        stack.append(node);

        callback(node, level);

        for (auto& input : getNodeInputs(node)) {
            if (!visit(input, stack, level + 1, callback))
                return false;
        }

        ASSERT(!stack.isEmpty());
        ASSERT(stack.last().ptr() == &node);

        stack.removeLast();
        return true;
    }

    UncheckedKeyHashMap<AtomString, Ref<NodeType>> m_sourceNodes;
    UncheckedKeyHashMap<AtomString, Ref<NodeType>> m_namedNodes;
    UncheckedKeyHashMap<Ref<NodeType>, NodeVector> m_nodeInputs;
    RefPtr<NodeType> m_lastNode;
};

using SVGFilterEffectsGraph = SVGFilterGraph<FilterEffect>;
using SVGFilterPrimitivesGraph = SVGFilterGraph<SVGFilterPrimitiveStandardAttributes>;

} // namespace WebCore
