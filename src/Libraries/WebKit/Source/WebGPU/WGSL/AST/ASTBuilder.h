/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/Vector.h>
#include <wtf/text/ParsingUtilities.h>

#define WGSL_AST_BUILDER_NODE(Node) \
protected: \
    Node(const Node&) = default; \
    Node(Node&&) = default; \
    Node& operator=(const Node&) = default; \
    Node& operator=(Node&&) = default; \
private: \
    friend class Builder; \
    friend ShaderModule;

namespace WGSL {

class ShaderModule;

namespace AST {

class Node;

class Builder {
    WTF_MAKE_NONCOPYABLE(Builder);

public:
    static constexpr size_t arenaSize = 0x4000;

    Builder() = default;
    Builder(Builder&&);
    ~Builder();

    template<typename T, typename... Arguments, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
    T& construct(Arguments&&... arguments)
    {
        constexpr size_t size = sizeof(T);
        constexpr size_t alignedSize = alignSize(size);
        static_assert(alignedSize <= arenaSize);
        if (UNLIKELY(m_arena.size() < alignedSize))
            allocateArena();

        auto* node = new (m_arena.data()) T(std::forward<Arguments>(arguments)...);
        skip(m_arena, alignedSize);
        m_nodes.append(node);
        return *node;
    }

    class State {
        friend Builder;
    private:
        State() = default;

        std::span<uint8_t> m_arena;
        unsigned m_numberOfArenas;
        unsigned m_numberOfNodes;
    };

    State saveCurrentState();
    void restore(State&&);

private:
    static constexpr size_t alignSize(size_t size)
    {
        return (size + sizeof(WTF::AllocAlignmentInteger) - 1) & ~(sizeof(WTF::AllocAlignmentInteger) - 1);
    }

    void allocateArena();

    std::span<uint8_t> m_arena { };
    uint8_t* m_arenaEnd { nullptr };
    Vector<FixedVector<uint8_t>> m_arenas;
    Vector<Node*> m_nodes;
};

} // namespace AST
} // namespace WGSL
