/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "config.h"

#if ENABLE(WEBASSEMBLY)

#include "WasmSourceMappingURLSectionParser.h"
#include "WasmTypeDefinitionInlines.h"

namespace JSC {
namespace Wasm {

auto SourceMappingURLSectionParser::parse() -> PartialResult
{
    uint32_t length;
    Name name;
    WASM_PARSER_FAIL_IF(!parseVarUInt32(length), "can't get source mapping URL length"_s);
    WASM_PARSER_FAIL_IF(!consumeUTF8String(name, length), "can't get source mapping URL of length "_s, length, " for payload "_s);
    m_info->sourceMappingURL = WTFMove(name);
    return { };
}

}
} // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
