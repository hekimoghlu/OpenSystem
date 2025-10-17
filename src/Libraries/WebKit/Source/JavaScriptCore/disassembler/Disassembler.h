/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

#include "JSCPtrTag.h"
#include "JSExportMacros.h"
#include <functional>
#include <wtf/CodePtr.h>
#include <wtf/PrintStream.h>
#include <wtf/text/CString.h>

namespace JSC {

template<PtrTag> class MacroAssemblerCodeRef;

#if ENABLE(DISASSEMBLER)
JS_EXPORT_PRIVATE bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>&, size_t, void* codeStart, void* codeEnd, const char* prefix, PrintStream&);
#else
inline bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>&, size_t, void*, void*, const char*, PrintStream&)
{
    return false;
}
#endif

inline bool tryToDisassemble(const CodePtr<DisassemblyPtrTag>& code, size_t size, const char* prefix, PrintStream& out)
{
    return tryToDisassemble(code, size, nullptr, nullptr, prefix, out);
}

// Prints either the disassembly, or a line of text indicating that disassembly failed and
// the range of machine code addresses.
void disassemble(const CodePtr<DisassemblyPtrTag>&, size_t, void* codeStart, void* codeEnd, const char* prefix, PrintStream& out);

// Asynchronous disassembly. This happens on another thread, and calls the provided
// callback when the disassembly is done.
void disassembleAsynchronously(
    const CString& header, const MacroAssemblerCodeRef<DisassemblyPtrTag>&, size_t, void* codeStart, void* codeEnd, const char* prefix);

JS_EXPORT_PRIVATE void waitForAsynchronousDisassembly();

void registerLabel(void* thunkAddress, CString&& label);
void registerLabel(void* thunkAddress, const char* label);
const char* labelFor(void* thunkAddress);

} // namespace JSC
