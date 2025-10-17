/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#if ENABLE(WEBASSEMBLY) && ENABLE(JIT)

#include "JITCompilation.h"
#include "WasmBinding.h"
#include "WasmJS.h"
#include <wtf/Bag.h>
#include <wtf/Expected.h>

namespace JSC {

class OptimizingCallLinkInfo;
class VM;

namespace Wasm {

class WasmToJSCallee;

Expected<MacroAssemblerCodeRef<WasmEntryPtrTag>, BindingFailure> wasmToJS(TypeIndex, unsigned importIndex);

void emitThrowWasmToJSException(CCallHelpers&, GPRReg wasmInstance, Wasm::ExceptionType);

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY) && ENABLE(JIT)
