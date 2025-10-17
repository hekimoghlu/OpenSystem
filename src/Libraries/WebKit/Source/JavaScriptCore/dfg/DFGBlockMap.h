/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#if ENABLE(DFG_JIT)

#include "DFGBasicBlock.h"

namespace JSC { namespace DFG {

class Graph;

template<typename T>
class BlockMap {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(BlockMap);
public:
    BlockMap()
    {
    }
    
    BlockMap(Graph&);
    
    BlockIndex size() const
    {
        return m_vector.size();
    }
    
    T& atIndex(BlockIndex blockIndex)
    {
        return m_vector[blockIndex];
    }
    
    const T& atIndex(BlockIndex blockIndex) const
    {
        return m_vector[blockIndex];
    }
    
    T& at(BlockIndex blockIndex)
    {
        return m_vector[blockIndex];
    }
    
    const T& at(BlockIndex blockIndex) const
    {
        return m_vector[blockIndex];
    }
    
    T& at(BasicBlock* block)
    {
        return m_vector[block->index];
    }
    
    const T& at(BasicBlock* block) const
    {
        return m_vector[block->index];
    }

    T& operator[](BlockIndex blockIndex)
    {
        return m_vector[blockIndex];
    }
    
    const T& operator[](BlockIndex blockIndex) const
    {
        return m_vector[blockIndex];
    }
    
    T& operator[](BasicBlock* block)
    {
        return m_vector[block->index];
    }
    
    const T& operator[](BasicBlock* block) const
    {
        return m_vector[block->index];
    }

private:
    Vector<T> m_vector;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename T>, BlockMap<T>);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
