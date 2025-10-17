/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include "ASTBuilder.h"

#include "ASTNode.h"
#include <wtf/FixedVector.h>

namespace WGSL::AST {

Builder::Builder(Builder&& other)
{
    m_arena = std::exchange(other.m_arena, { });
    m_arenas = WTFMove(other.m_arenas);
    m_nodes = WTFMove(other.m_nodes);
}

Builder::~Builder()
{
    size_t size = m_nodes.size();
    for (size_t i = 0; i < size; ++i)
        m_nodes[i]->~Node();
}

void Builder::allocateArena()
{
    m_arenas.append(FixedVector<uint8_t>(arenaSize));
    m_arena = m_arenas.last().mutableSpan();
}

auto Builder::saveCurrentState() -> State
{
    State state;
    state.m_arena = m_arena;
    state.m_numberOfArenas = m_arenas.size();
    state.m_numberOfNodes = m_nodes.size();
    allocateArena();
    return state;
}

void Builder::restore(State&& state)
{
    for (size_t i = state.m_numberOfNodes; i < m_nodes.size(); ++i)
        m_nodes[i]->~Node();
    m_nodes.shrink(state.m_numberOfNodes);
    m_arena = state.m_arena;
    m_arenas.shrink(state.m_numberOfArenas);
}

} // namespace WGSL::AST
