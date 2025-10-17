/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

#include "ImmutableNFA.h"
#include "MutableRangeList.h"
#include <wtf/HashSet.h>

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore {

namespace ContentExtensions {

// A ImmutableNFANodeBuilder let you build an NFA node by adding states and linking with other nodes.
// When a builder is destructed, all its properties are finalized into the NFA. Using the NFA with a live
// builder results in undefined behavior.
template <typename CharacterType, typename ActionType>
class ImmutableNFANodeBuilder {
    typedef ImmutableNFA<CharacterType, ActionType> TypedImmutableNFA;
    typedef UncheckedKeyHashSet<uint32_t, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> TargetSet;
public:
    ImmutableNFANodeBuilder() { }

    ImmutableNFANodeBuilder(TypedImmutableNFA& immutableNFA)
        : m_immutableNFA(&immutableNFA)
        , m_finalized(false)
    {
        m_nodeId = m_immutableNFA->nodes.size();
        m_immutableNFA->nodes.append(ImmutableNFANode());
    }

    ImmutableNFANodeBuilder(ImmutableNFANodeBuilder&& other)
        : m_immutableNFA(other.m_immutableNFA)
        , m_ranges(WTFMove(other.m_ranges))
        , m_epsilonTransitionTargets(WTFMove(other.m_epsilonTransitionTargets))
        , m_actions(WTFMove(other.m_actions))
        , m_nodeId(other.m_nodeId)
        , m_finalized(other.m_finalized)
    {
        other.m_immutableNFA = nullptr;
        other.m_finalized = true;
    }

    ~ImmutableNFANodeBuilder()
    {
        if (!m_finalized)
            finalize();
    }

    bool isValid() const
    {
        return !!m_immutableNFA;
    }

    uint32_t nodeId() const
    {
        ASSERT(isValid());
        return m_nodeId;
    }

    struct TrivialRange {
        CharacterType first;
        CharacterType last;
    };
    
    struct FakeRangeIterator {
        CharacterType first() const { return range.first; }
        CharacterType last() const { return range.last; }
        uint32_t data() const { return targetId; }
        bool operator==(const FakeRangeIterator& other) const
        {
            return this->isEnd == other.isEnd;
        }
        FakeRangeIterator operator++()
        {
            isEnd = true;
            return *this;
        }
        
        TrivialRange range;
        uint32_t targetId;
        bool isEnd;
    };

    void addTransition(CharacterType first, CharacterType last, uint32_t targetNodeId)
    {
        ASSERT(!m_finalized);
        ASSERT(m_immutableNFA);

        struct Converter {
            TargetSet convert(uint32_t target)
            {
                return TargetSet({ target });
            }
            void extend(TargetSet& existingTargets, uint32_t target)
            {
                existingTargets.add(target);
            }
        };
        
        Converter converter;
        m_ranges.extend(FakeRangeIterator { { first, last }, targetNodeId, false }, FakeRangeIterator { { 0, 0 }, targetNodeId, true }, converter);
    }

    void addEpsilonTransition(const ImmutableNFANodeBuilder& target)
    {
        ASSERT(m_immutableNFA == target.m_immutableNFA);
        addEpsilonTransition(target.m_nodeId);
    }

    void addEpsilonTransition(uint32_t targetNodeId)
    {
        ASSERT(!m_finalized);
        ASSERT(m_immutableNFA);
        m_epsilonTransitionTargets.add(targetNodeId);
    }

    template<typename ActionIterator>
    void setActions(ActionIterator begin, ActionIterator end)
    {
        ASSERT(!m_finalized);
        ASSERT(m_immutableNFA);

        m_actions.add(begin, end);
    }

    ImmutableNFANodeBuilder& operator=(ImmutableNFANodeBuilder&& other)
    {
        if (!m_finalized)
            finalize();

        m_immutableNFA = other.m_immutableNFA;
        m_ranges = WTFMove(other.m_ranges);
        m_epsilonTransitionTargets = WTFMove(other.m_epsilonTransitionTargets);
        m_actions = WTFMove(other.m_actions);
        m_nodeId = other.m_nodeId;
        m_finalized = other.m_finalized;

        other.m_immutableNFA = nullptr;
        other.m_finalized = true;
        return *this;
    }

private:
    void finalize()
    {
        ASSERT_WITH_MESSAGE(!m_finalized, "The API contract is that the builder can only be finalized once.");
        m_finalized = true;
        ImmutableNFANode& immutableNFANode = m_immutableNFA->nodes[m_nodeId];
        sinkActions(immutableNFANode);
        sinkEpsilonTransitions(immutableNFANode);
        sinkTransitions(immutableNFANode);
    }

    void sinkActions(ImmutableNFANode& immutableNFANode)
    {
        unsigned actionStart = m_immutableNFA->actions.size();
        for (const ActionType& action : m_actions)
            m_immutableNFA->actions.append(action);
        unsigned actionEnd = m_immutableNFA->actions.size();
        immutableNFANode.actionStart = actionStart;
        immutableNFANode.actionEnd = actionEnd;
    }

    void sinkTransitions(ImmutableNFANode& immutableNFANode)
    {
        unsigned transitionsStart = m_immutableNFA->transitions.size();
        for (const auto& range : m_ranges) {
            unsigned targetsStart = m_immutableNFA->targets.size();
            for (uint32_t target : range.data)
                m_immutableNFA->targets.append(target);
            unsigned targetsEnd = m_immutableNFA->targets.size();

            m_immutableNFA->transitions.append(ImmutableRange<CharacterType> { targetsStart, targetsEnd, range.first, range.last });
        }
        unsigned transitionsEnd = m_immutableNFA->transitions.size();

        immutableNFANode.rangesStart = transitionsStart;
        immutableNFANode.rangesEnd = transitionsEnd;
    }

    void sinkEpsilonTransitions(ImmutableNFANode& immutableNFANode)
    {
        unsigned start = m_immutableNFA->epsilonTransitionsTargets.size();
        for (uint32_t target : m_epsilonTransitionTargets)
            m_immutableNFA->epsilonTransitionsTargets.append(target);
        unsigned end = m_immutableNFA->epsilonTransitionsTargets.size();

        immutableNFANode.epsilonTransitionTargetsStart = start;
        immutableNFANode.epsilonTransitionTargetsEnd = end;
    }

    TypedImmutableNFA* m_immutableNFA { nullptr };
    MutableRangeList<CharacterType, TargetSet> m_ranges;
    TargetSet m_epsilonTransitionTargets;
    UncheckedKeyHashSet<ActionType, IntHash<ActionType>, WTF::UnsignedWithZeroKeyHashTraits<ActionType>> m_actions;
    uint32_t m_nodeId;
    bool m_finalized { true };
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
