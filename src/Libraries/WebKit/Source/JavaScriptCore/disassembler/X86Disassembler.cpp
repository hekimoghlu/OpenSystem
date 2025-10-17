/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#include "Disassembler.h"

#if ENABLE(ZYDIS)

#include "AssemblyComments.h"
#include "MacroAssemblerCodeRef.h"
#include "Zydis.h"
#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>& codePtr, size_t size, void*, void*, const char* prefix, PrintStream& out)
{
    ZydisDecoder decoder;
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_ADDRESS_WIDTH_64);

    ZydisFormatter formatter;
    ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_ATT);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_FORCE_SIZE, ZYAN_TRUE);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_HEX_UPPERCASE, ZYAN_FALSE);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_ADDR_PADDING_ABSOLUTE, ZYDIS_PADDING_DISABLED);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_ADDR_PADDING_RELATIVE, ZYDIS_PADDING_DISABLED);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_DISP_PADDING, ZYDIS_PADDING_DISABLED);
    ZydisFormatterSetProperty(&formatter, ZYDIS_FORMATTER_PROP_IMM_PADDING, ZYDIS_PADDING_DISABLED);

    const auto* data = codePtr.dataLocation<unsigned char*>();
    ZyanUSize offset = 0;
    ZydisDecodedInstruction instruction;
    char formatted[1024];
    while (ZYAN_SUCCESS(ZydisDecoderDecodeBuffer(&decoder, data + offset, size - offset, &instruction))) {
        if (ZYAN_SUCCESS(ZydisFormatterFormatInstruction(&formatter, &instruction, formatted, sizeof(formatted), std::bit_cast<unsigned long long>(data + offset))))
            out.printf("%s%#16llx: %s", prefix, static_cast<unsigned long long>(std::bit_cast<uintptr_t>(data + offset)), formatted);
        else
            out.printf("%s%#16llx: failed-to-format", prefix, static_cast<unsigned long long>(std::bit_cast<uintptr_t>(data + offset)));
        if (auto str = AssemblyCommentRegistry::singleton().comment(reinterpret_cast<void*>(std::bit_cast<uintptr_t>(data + offset))))
            out.printf("; %s\n", str->ascii().data());
        else
            out.printf("\n");
        offset += instruction.length;
    }

    return true;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(ZYDIS)
