/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "AssemblyComments.h"
#include "Disassembler.h"

#if ENABLE(ARM64_DISASSEMBLER)

#include "A64DOpcode.h"
#include "MacroAssemblerCodeRef.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>& codePtr, size_t size, void* codeStart, void* codeEnd, const char* prefix, PrintStream& out)
{
    uint32_t* currentPC = codePtr.untaggedPtr<uint32_t*>();
    size_t byteCount = size;

    uint32_t* armCodeStart = std::bit_cast<uint32_t*>(codeStart);
    uint32_t* armCodeEnd = std::bit_cast<uint32_t*>(codeEnd);
    A64DOpcode arm64Opcode(armCodeStart, armCodeEnd);

    unsigned pcOffset = (currentPC - armCodeStart) * sizeof(uint32_t);
    char pcInfo[25];
    while (byteCount) {
        if (codeStart)
            snprintf(pcInfo, sizeof(pcInfo) - 1, "<%u> %#llx", pcOffset, static_cast<unsigned long long>(std::bit_cast<uintptr_t>(currentPC)));
        else
            snprintf(pcInfo, sizeof(pcInfo) - 1, "%#llx", static_cast<unsigned long long>(std::bit_cast<uintptr_t>(currentPC)));
        out.printf("%s%24s: %s", prefix, pcInfo, arm64Opcode.disassemble(currentPC));
        if (auto str = AssemblyCommentRegistry::singleton().comment(currentPC))
            out.printf("; %s\n", str->ascii().data());
        else
            out.printf("\n");
        pcOffset += sizeof(uint32_t);
        currentPC++;
        byteCount -= sizeof(uint32_t);
    }

    return true;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(ARM64_DISASSEMBLER)
