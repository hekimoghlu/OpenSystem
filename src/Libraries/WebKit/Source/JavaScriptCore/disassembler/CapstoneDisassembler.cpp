/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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

#if ENABLE(DISASSEMBLER) && USE(CAPSTONE)

#include "MacroAssemblerCodeRef.h"
#include "Options.h"
#include <capstone/capstone.h>

namespace JSC {

bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>& codePtr, size_t size, void*, void*, const char* prefix, PrintStream& out)
{
    csh handle;
    cs_insn* instructions;

#if CPU(X86_64)
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &handle) != CS_ERR_OK)
        return false;
#elif CPU(ARM_THUMB2)
    if (cs_open(CS_ARCH_ARM, CS_MODE_THUMB, &handle) != CS_ERR_OK)
        return false;
    if (cs_option(handle, CS_OPT_SYNTAX, CS_OPT_SYNTAX_NOREGNAME) != CS_ERR_OK)
        return false;
#elif CPU(ARM64)
    if (cs_open(CS_ARCH_ARM64, CS_MODE_ARM, &handle) != CS_ERR_OK)
        return false;
#else
    return false;
#endif

#if CPU(X86_64)
    if (cs_option(handle, CS_OPT_SYNTAX, CS_OPT_SYNTAX_ATT) != CS_ERR_OK) {
        cs_close(&handle);
        return false;
    }
#endif

    size_t count = cs_disasm(handle, codePtr.dataLocation<unsigned char*>(), size, codePtr.dataLocation<uintptr_t>(), 0, &instructions);
    if (count > 0) {
        for (size_t i = 0; i < count; ++i) {
            auto& instruction = instructions[i];
            out.printf("%s%#16llx: %s %s", prefix, static_cast<unsigned long long>(instruction.address), instruction.mnemonic, instruction.op_str);
            if (auto str = AssemblyCommentRegistry::singleton().comment(reinterpret_cast<void *>(static_cast<uintptr_t>(instruction.address))))
                out.printf("; %s\n", str->ascii().data());
            else
                out.printf("\n");
        }
        cs_free(instructions, count);
    }
    cs_close(&handle);
    return true;
}

} // namespace JSC

#endif // ENABLE(DISASSEMBLER) && USE(CAPSTONE)
