/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#include "MacroAssemblerCodeRef.h"

#include "CodeBlock.h"
#include "Disassembler.h"
#include "JITCode.h"
#include "JSCPtrTag.h"
#include "WasmCompilationMode.h"
#include <wtf/StringPrintStream.h>

namespace JSC {

bool MacroAssemblerCodeRefBase::tryToDisassemble(CodePtr<DisassemblyPtrTag> codePtr, size_t size, const char* prefix, PrintStream& out)
{
    return JSC::tryToDisassemble(codePtr, size, prefix, out);
}

bool MacroAssemblerCodeRefBase::tryToDisassemble(CodePtr<DisassemblyPtrTag> codePtr, size_t size, const char* prefix)
{
    return tryToDisassemble(codePtr, size, prefix, WTF::dataFile());
}

CString MacroAssemblerCodeRefBase::disassembly(CodePtr<DisassemblyPtrTag> codePtr, size_t size)
{
    StringPrintStream out;
    if (!tryToDisassemble(codePtr, size, "", out))
        return CString();
    return out.toCString();
}

bool shouldDumpDisassemblyFor(CodeBlock* codeBlock)
{
    if (codeBlock && JSC::JITCode::isOptimizingJIT(codeBlock->jitType()) && Options::dumpDFGDisassembly())
        return true;
    return Options::dumpDisassembly();
}

bool shouldDumpDisassemblyFor(Wasm::CompilationMode mode)
{
    if (Options::asyncDisassembly() || Options::dumpDisassembly() || Options::dumpWasmDisassembly())
        return true;
    if (Wasm::isAnyBBQ(mode))
        return Options::dumpBBQDisassembly();
    if (Wasm::isAnyOMG(mode))
        return Options::dumpOMGDisassembly();
    return false;
}

} // namespace JSC

