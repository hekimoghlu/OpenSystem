/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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

#if ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)

#include "B3Common.h"
#include "B3Procedure.h"
#include "CCallHelpers.h"
#include "JITCompilation.h"
#include "JITOpaqueByproducts.h"
#include "PCToCodeOriginMap.h"
#include "WasmBBQDisassembler.h"
#include "WasmCompilationMode.h"
#include "WasmJS.h"
#include "WasmMemory.h"
#include "WasmModuleInformation.h"
#include "WasmOpcodeOrigin.h"
#include "WasmTierUpCount.h"
#include <wtf/Box.h>
#include <wtf/Expected.h>
#include <wtf/SegmentedVector.h>

namespace JSC {

#if !ENABLE(B3_JIT)
namespace B3 {

class Procedure { };

}
#endif

namespace Wasm {

class BBQDisassembler;
class CalleeGroup;
class MemoryInformation;
class OptimizingJITCallee;
class TierUpCount;

class OMGOrigin {
public:
    friend bool operator==(const OMGOrigin&, const OMGOrigin&) = default;

    CallSiteIndex m_callSiteIndex { };
    OpcodeOrigin m_opcodeOrigin { };
};

struct CompilationContext {
    std::unique_ptr<CCallHelpers> jsEntrypointJIT;
    std::unique_ptr<CCallHelpers> wasmEntrypointJIT;
    std::unique_ptr<OpaqueByproducts> wasmEntrypointByproducts;
    std::unique_ptr<B3::Procedure> procedure;
    std::unique_ptr<BBQDisassembler> bbqDisassembler;
    Box<PCToCodeOriginMap> pcToCodeOriginMap;
    Box<PCToCodeOriginMapBuilder> pcToCodeOriginMapBuilder;
    Vector<CCallHelpers::Label> catchEntrypoints;
    SegmentedVector<OMGOrigin> origins;
};

} // namespace Wasm

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)
