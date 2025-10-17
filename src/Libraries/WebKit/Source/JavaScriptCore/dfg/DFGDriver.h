/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#include "DFGPlan.h"

namespace JSC {

class CodeBlock;
class JITCode;
class VM;
enum class JITCompilationMode;

namespace DFG {

JS_EXPORT_PRIVATE unsigned getNumCompilations();

// If the worklist is non-null, we do a concurrent compile. Otherwise we do a synchronous
// compile. Even if we do a synchronous compile, we call the callback with the result.
CompilationResult compile(
    VM&, CodeBlock*, CodeBlock* profiledDFGCodeBlock, JITCompilationMode,
    BytecodeIndex osrEntryBytecodeIndex, Operands<std::optional<JSValue>>&& mustHandleValues,
    Ref<DeferredCompilationCallback>&&);

} } // namespace JSC::DFG
