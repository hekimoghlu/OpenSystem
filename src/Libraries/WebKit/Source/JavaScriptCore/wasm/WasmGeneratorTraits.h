/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#if ENABLE(WEBASSEMBLY)

#include "BytecodeStructs.h"
#include "Instruction.h"

namespace JSC { namespace Wasm {

struct GeneratorTraits {
    using OpcodeTraits = WasmOpcodeTraits;
    using OpcodeID = WasmOpcodeID;
    using OpNop = WasmNop;
    using CodeBlock = std::unique_ptr<FunctionCodeBlockGenerator>;
    using InstructionType = WasmInstruction;
    static constexpr OpcodeID opcodeForDisablingOptimizations = wasm_unreachable;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
