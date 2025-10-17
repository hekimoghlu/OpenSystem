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
#include "config.h"
#include "DeferredSourceDump.h"

#include "CodeBlock.h"
#include "CodeBlockWithJITType.h"
#include "StrongInlines.h"

namespace JSC {

DeferredSourceDump::DeferredSourceDump(CodeBlock* codeBlock)
    : m_codeBlock(codeBlock->vm(), codeBlock)
    , m_rootJITType(JITType::None)
{
}

DeferredSourceDump::DeferredSourceDump(CodeBlock* codeBlock, CodeBlock* rootCodeBlock, JITType rootJITType, BytecodeIndex callerBytecodeIndex)
    : m_codeBlock(codeBlock->vm(), codeBlock)
    , m_rootCodeBlock(codeBlock->vm(), rootCodeBlock)
    , m_rootJITType(rootJITType)
    , m_callerBytecodeIndex(callerBytecodeIndex)
{
}

void DeferredSourceDump::dump()
{
    bool isInlinedFrame = !!m_rootCodeBlock;
    if (isInlinedFrame)
        dataLog("Inlined ");
    else
        dataLog("Compiled ");
    dataLog(*m_codeBlock);

    if (isInlinedFrame)
        dataLog(" at ", CodeBlockWithJITType(*m_rootCodeBlock, m_rootJITType), " ", m_callerBytecodeIndex);

    dataLog("\n'''");
    m_codeBlock->dumpSource();
    dataLog("'''\n");
}

} // namespace JSC
