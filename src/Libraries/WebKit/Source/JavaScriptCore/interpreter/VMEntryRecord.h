/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

#include "GPRInfo.h"

namespace JSC {

struct EntryFrame;
class CallFrame;
class JSGlobalObject;
class JSObject;
class VM;

struct VMEntryRecord {
    /*
     * This record stored in a vmEntryTo{JavaScript,Host} allocated frame. It is allocated on the stack
     * after callee save registers where local variables would go.
     */
    VM* const m_vm;
    CallFrame* const m_prevTopCallFrame;
    EntryFrame* const m_prevTopEntryFrame;

#if !ENABLE(C_LOOP) && NUMBER_OF_CALLEE_SAVES_REGISTERS > 0
    CPURegister calleeSaveRegistersBuffer[NUMBER_OF_CALLEE_SAVES_REGISTERS];
#elif ENABLE(C_LOOP)
    CPURegister calleeSaveRegistersBuffer[1];
#endif

    CallFrame* prevTopCallFrame() { return m_prevTopCallFrame; }
    SUPPRESS_ASAN CallFrame* unsafePrevTopCallFrame() { return m_prevTopCallFrame; }

    EntryFrame* prevTopEntryFrame() { return m_prevTopEntryFrame; }
    SUPPRESS_ASAN EntryFrame* unsafePrevTopEntryFrame() { return m_prevTopEntryFrame; }
};

extern "C" VMEntryRecord* SYSV_ABI vmEntryRecord(EntryFrame*);

} // namespace JSC
