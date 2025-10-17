/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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

#include "RenderBlock.h"
#include "RenderElementInlines.h"

namespace WebCore {

class LogicalSelectionOffsetCaches {
public:
    class ContainingBlockInfo {
    public:
        ContainingBlockInfo()
            : m_hasFloatsOrFragmentedFlows(false)
            , m_cachedLogicalLeftSelectionOffset(false)
            , m_cachedLogicalRightSelectionOffset(false)
        { }

        void setBlock(RenderBlock* block, const LogicalSelectionOffsetCaches* cache, bool parentCacheHasFloatsOrFragmentedFlows = false)
        {
            m_block = block;
            bool blockHasFloatsOrFragmentedFlows = m_block ? (m_block->containsFloats() || m_block->enclosingFragmentedFlow()) : false;
            m_hasFloatsOrFragmentedFlows = parentCacheHasFloatsOrFragmentedFlows || m_hasFloatsOrFragmentedFlows || blockHasFloatsOrFragmentedFlows;
            m_cache = cache;
            m_cachedLogicalLeftSelectionOffset = false;
            m_cachedLogicalRightSelectionOffset = false;
        }

        LayoutUnit logicalLeftSelectionOffset(RenderBlock& rootBlock, LayoutUnit position) const
        {
            ASSERT(m_cache);
            if (m_hasFloatsOrFragmentedFlows || !m_cachedLogicalLeftSelectionOffset) {
                m_cachedLogicalLeftSelectionOffset = true;
                m_logicalLeftSelectionOffset = m_block ? m_block->logicalLeftSelectionOffset(rootBlock, position, *m_cache) : 0_lu;
            } else
                ASSERT(m_logicalLeftSelectionOffset == (m_block ? m_block->logicalLeftSelectionOffset(rootBlock, position, *m_cache) : 0_lu));
            return m_logicalLeftSelectionOffset;
        }

        LayoutUnit logicalRightSelectionOffset(RenderBlock& rootBlock, LayoutUnit position) const
        {
            ASSERT(m_cache);
            if (m_hasFloatsOrFragmentedFlows || !m_cachedLogicalRightSelectionOffset) {
                m_cachedLogicalRightSelectionOffset = true;
                m_logicalRightSelectionOffset = m_block ? m_block->logicalRightSelectionOffset(rootBlock, position, *m_cache) : 0_lu;
            } else
                ASSERT(m_logicalRightSelectionOffset == (m_block ? m_block->logicalRightSelectionOffset(rootBlock, position, *m_cache) : 0_lu));
            return m_logicalRightSelectionOffset;
        }

        RenderBlock* block() const { return m_block; }
        const LogicalSelectionOffsetCaches* cache() const { return m_cache; }
        bool hasFloatsOrFragmentedFlows() const { return m_hasFloatsOrFragmentedFlows; }

    private:
        RenderBlock* m_block { nullptr };
        const LogicalSelectionOffsetCaches* m_cache { nullptr };
        bool m_hasFloatsOrFragmentedFlows : 1;
        mutable bool m_cachedLogicalLeftSelectionOffset : 1;
        mutable bool m_cachedLogicalRightSelectionOffset : 1;
        mutable LayoutUnit m_logicalLeftSelectionOffset;
        mutable LayoutUnit m_logicalRightSelectionOffset;
        
    };

    explicit LogicalSelectionOffsetCaches(RenderBlock& rootBlock)
    {
#if ENABLE(TEXT_SELECTION)
        // FIXME: We should either move this assertion to the caller (if applicable) or structure the code
        // such that we can remove this assertion.
        ASSERT(rootBlock.isSelectionRoot());
#endif
        // LogicalSelectionOffsetCaches should not be used on an orphaned tree.
        m_containingBlockForFixedPosition.setBlock(RenderObject::containingBlockForPositionType(PositionType::Fixed, rootBlock), nullptr);
        m_containingBlockForAbsolutePosition.setBlock(RenderObject::containingBlockForPositionType(PositionType::Absolute, rootBlock), nullptr);
        m_containingBlockForInflowPosition.setBlock(RenderObject::containingBlockForPositionType(PositionType::Static, rootBlock), nullptr);
    }

    LogicalSelectionOffsetCaches(RenderBlock& block, const LogicalSelectionOffsetCaches& cache)
        : m_containingBlockForFixedPosition(cache.m_containingBlockForFixedPosition)
        , m_containingBlockForAbsolutePosition(cache.m_containingBlockForAbsolutePosition)
    {
        if (block.canContainFixedPositionObjects())
            m_containingBlockForFixedPosition.setBlock(&block, &cache, cache.m_containingBlockForFixedPosition.hasFloatsOrFragmentedFlows());

        if (block.canContainAbsolutelyPositionedObjects() && !block.isRenderInline() && !block.isAnonymousBlock())
            m_containingBlockForAbsolutePosition.setBlock(&block, &cache, cache.m_containingBlockForAbsolutePosition.hasFloatsOrFragmentedFlows());

        m_containingBlockForInflowPosition.setBlock(&block, &cache, cache.m_containingBlockForInflowPosition.hasFloatsOrFragmentedFlows());
    }

    const ContainingBlockInfo& containingBlockInfo(RenderBlock& block) const
    {
        auto position = block.style().position();
        if (position == PositionType::Fixed) {
            ASSERT(block.containingBlock() == m_containingBlockForFixedPosition.block());
            return m_containingBlockForFixedPosition;
        }
        if (position == PositionType::Absolute) {
            ASSERT(block.containingBlock() == m_containingBlockForAbsolutePosition.block());
            return m_containingBlockForAbsolutePosition;
        }
        ASSERT(block.containingBlock() == m_containingBlockForInflowPosition.block());
        return m_containingBlockForInflowPosition;
    }

private:
    ContainingBlockInfo m_containingBlockForFixedPosition;
    ContainingBlockInfo m_containingBlockForAbsolutePosition;
    ContainingBlockInfo m_containingBlockForInflowPosition;
};

} // namespace WebCore
