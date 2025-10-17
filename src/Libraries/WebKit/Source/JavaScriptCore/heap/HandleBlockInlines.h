/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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

#include "HandleBlock.h"
#include <wtf/FastMalloc.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

inline HandleBlock* HandleBlock::create(HandleSet* handleSet)
{
    return new (NotNull, fastAlignedMalloc(blockSize, blockSize)) HandleBlock(handleSet);
}

inline void HandleBlock::destroy(HandleBlock* block)
{
    block->~HandleBlock();
    fastAlignedFree(block);
}

inline HandleBlock::HandleBlock(HandleSet* handleSet)
    : DoublyLinkedListNode<HandleBlock>()
    , m_handleSet(handleSet)
{
}

inline char* HandleBlock::payloadEnd()
{
    return reinterpret_cast<char*>(this) + blockSize;
}

inline char* HandleBlock::payload()
{
    return reinterpret_cast<char*>(this) + WTF::roundUpToMultipleOf<sizeof(double)>(sizeof(HandleBlock));
}

inline HandleNode* HandleBlock::nodes()
{
    return reinterpret_cast_ptr<HandleNode*>(payload());
}

inline HandleNode* HandleBlock::nodeAtIndex(unsigned i)
{
    ASSERT(i < nodeCapacity());
    return &nodes()[i];
}

inline unsigned HandleBlock::nodeCapacity()
{
    return (payloadEnd() - payload()) / sizeof(HandleNode);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
