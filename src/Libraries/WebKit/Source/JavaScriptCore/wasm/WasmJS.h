/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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

#include "WasmExceptionType.h"
#include "WasmMemory.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>

#include <memory>

namespace JSC {

class CCallHelpers;
class CallFrame;

namespace Wasm {

struct CompilationContext;
struct InternalFunction;
struct ModuleInformation;
class TypeDefinition;
struct UnlinkedWasmToWasmCall;

// Create wrapper code to call from JS -> WebAssembly.
using CreateJSWrapper = WTF::Function<std::unique_ptr<InternalFunction>(CCallHelpers&, const TypeDefinition&, Vector<UnlinkedWasmToWasmCall>*, const ModuleInformation&, MemoryMode, uint32_t)>;

// Called as soon as an exception is detected. The return value is the PC to continue at.
using ThrowWasmException = void* (*)(CallFrame*, Wasm::ExceptionType, JSWebAssemblyInstance*);

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
