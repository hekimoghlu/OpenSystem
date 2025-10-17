/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include "OpcodeSize.h"
#include <wtf/StdLibExtras.h>

namespace JSC {

class CallFrame;
class VM;
template<PtrTag> class MacroAssemblerCodeRef;

template<typename> struct BaseInstruction;
struct JSOpcodeTraits;
struct WasmOpcodeTraits;

using JSInstruction = BaseInstruction<JSOpcodeTraits>;
using WasmInstruction = BaseInstruction<WasmOpcodeTraits>;

namespace LLInt {

// Gives you a PC that you can tell the interpreter to go to, which when advanced
// between 1 and 9 slots will give you an "instruction" that threads to the
// interpreter's exception handler.
JSInstruction* returnToThrow(VM&);
WasmInstruction* wasmReturnToThrow(VM&);

// Use this when you're throwing to a call thunk.
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> callToThrow(VM&);

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleUncaughtException(VM&);
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleCatch(OpcodeSize);

#if ENABLE(WEBASSEMBLY)
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatch(OpcodeSize);
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatchAll(OpcodeSize);
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmTryTable(WasmOpcodeID, OpcodeSize);
#endif // ENABLE(WEBASSEMBLY)

} } // namespace JSC::LLInt
