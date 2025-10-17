/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

#include "WasmFormat.h"
#include <wtf/Expected.h>
#include <wtf/text/WTFString.h>

namespace JSC { namespace Wasm {

class FunctionCodeBlockGenerator;
class TypeDefinition;
struct ModuleInformation;

enum class UseDefaultValue : bool { No, Yes };
enum class ArrayGetKind : unsigned { New, NewDefault, NewFixed };

Expected<std::unique_ptr<FunctionCodeBlockGenerator>, String> parseAndCompileBytecode(std::span<const uint8_t>, const TypeDefinition&, ModuleInformation&, FunctionCodeIndex functionIndex);

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
