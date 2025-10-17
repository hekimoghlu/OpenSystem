/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#include <cstdint>

namespace JSC {

namespace Wasm {

// These limits are arbitrary except that they match the limits imposed
// by other browsers' implementation of WebAssembly. It is desirable for
// us to accept at least the same inputs.

constexpr size_t maxTypes = 1000000;
constexpr size_t maxFunctions = 1000000;
constexpr size_t maxImports = 100000;
constexpr size_t maxExports = 100000;
constexpr size_t maxExceptions = 100000;
constexpr size_t maxGlobals = 1000000;
constexpr size_t maxDataSegments = 100000;
constexpr size_t maxStructFieldCount = 10000;
constexpr size_t maxArrayNewFixedArgs = 10000;
constexpr size_t maxRecursionGroupCount = 1000000;
constexpr size_t maxNumberOfRecursionGroups = 1000000;
constexpr size_t maxSubtypeSupertypeCount = 1;
constexpr size_t maxSubtypeDepth = 63;

constexpr size_t maxStringSize = 100000;
constexpr size_t maxModuleSize = 1024 * 1024 * 1024;
constexpr size_t maxFunctionSize = 7654321;
constexpr size_t maxFunctionLocals = 50000;
constexpr size_t maxFunctionParams = 1000;
constexpr size_t maxFunctionReturns = 1000;

constexpr size_t maxTableEntries = 10000000;
constexpr unsigned maxTables = 1000000;

// Limit of GC arrays in bytes. This is not included in the limits in the
// JS API spec, but we set a limit to avoid complicated boundary conditions.
constexpr size_t maxArraySizeInBytes = 1 << 30;

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
