/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

#if ENABLE(WEBASSEMBLY_OMGJIT)

#include "B3Common.h"
#include "B3Procedure.h"
#include "CCallHelpers.h"
#include "JITCompilation.h"
#include "JITOpaqueByproducts.h"
#include "PCToCodeOriginMap.h"
#include "WasmBBQDisassembler.h"
#include "WasmCompilationContext.h"
#include "WasmCompilationMode.h"
#include "WasmJS.h"
#include "WasmMemory.h"
#include "WasmModuleInformation.h"
#include "WasmTierUpCount.h"
#include <wtf/Box.h>
#include <wtf/Expected.h>

extern "C" void SYSV_ABI dumpProcedure(void*);

namespace JSC {

namespace Wasm {

Expected<std::unique_ptr<InternalFunction>, String> parseAndCompileOMG(CompilationContext&, OptimizingJITCallee&, const FunctionData&, const TypeDefinition&, Vector<UnlinkedWasmToWasmCall>&, CalleeGroup&, const ModuleInformation&, MemoryMode, CompilationMode, FunctionCodeIndex functionIndex, std::optional<bool> hasExceptionHandlers, uint32_t loopIndexForOSREntry);

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)
