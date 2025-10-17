/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

#include "CPU.h"
#include "MacroAssembler.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class CCallHelpers;
class LLIntOffsetsExtractor;

class BasicBlockLocation {
    WTF_MAKE_TZONE_ALLOCATED(BasicBlockLocation);
public:
    typedef std::pair<int, int> Gap;

    BasicBlockLocation(int startOffset = -1, int endOffset = -1);

    int startOffset() const { return m_startOffset; }
    int endOffset() const { return m_endOffset; }
    void setStartOffset(int startOffset) { m_startOffset = startOffset; }
    void setEndOffset(int endOffset) { m_endOffset = endOffset; }
    bool hasExecuted() const { return m_executionCount > 0; }
    size_t executionCount() const { return m_executionCount; }
    void insertGap(int, int);
    Vector<Gap> getExecutedRanges() const;
    JS_EXPORT_PRIVATE void dumpData() const;
#if ENABLE(JIT)
#if USE(JSVALUE64)
    void emitExecuteCode(CCallHelpers&) const;
#else
    void emitExecuteCode(CCallHelpers&, MacroAssembler::RegisterID scratch) const;
#endif
#endif

private:
    friend class LLIntOffsetsExtractor;

    int m_startOffset;
    int m_endOffset;
    Vector<Gap> m_gaps;
    UCPURegister m_executionCount;
};

} // namespace JSC
