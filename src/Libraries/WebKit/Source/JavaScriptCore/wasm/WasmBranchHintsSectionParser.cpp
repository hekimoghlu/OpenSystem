/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#include "WasmBranchHintsSectionParser.h"
#include "WasmTypeDefinitionInlines.h"

namespace JSC {
namespace Wasm {

auto BranchHintsSectionParser::parse() -> PartialResult
{
    uint32_t functionCount;
    int64_t previousFunctionIndex = -1;
    WASM_PARSER_FAIL_IF(!parseVarUInt32(functionCount), "can't get function count"_s);

    for (uint32_t i = 0; i < functionCount; ++i) {
        uint32_t functionIndex;
        uint32_t hintCount;
        WASM_PARSER_FAIL_IF(!parseVarUInt32(functionIndex), "can't get function index for function "_s, i);
        WASM_PARSER_FAIL_IF(static_cast<int64_t>(functionIndex) < previousFunctionIndex, "invalid function index "_s, functionIndex, " for function "_s, i);

        previousFunctionIndex = functionIndex;

        WASM_PARSER_FAIL_IF(!parseVarUInt32(hintCount), "can't get number of hints for function "_s, i);

        if (!hintCount)
            continue;

        int64_t previousBranchOffset = -1;
        BranchHintMap branchHintsForFunction;
        for (uint32_t j = 0; j < hintCount; ++j) {
            uint32_t branchOffset;
            WASM_PARSER_FAIL_IF(!parseVarUInt32(branchOffset), "can't get branch offset for hint "_s, j);
            WASM_PARSER_FAIL_IF(static_cast<int64_t>(branchOffset) < previousBranchOffset
                || !m_info->branchHints.isValidKey(branchOffset), "invalid branch offset "_s, branchOffset, " for hint "_s, j);

            previousBranchOffset = branchOffset;

            uint32_t payloadSize;
            WASM_PARSER_FAIL_IF(!parseVarUInt32(payloadSize), "can't get payload size for hint "_s, j);
            WASM_PARSER_FAIL_IF(payloadSize != 0x1, "invalid payload size for hint "_s, j);

            uint8_t parsedBranchHint;
            WASM_PARSER_FAIL_IF(!parseVarUInt1(parsedBranchHint), "can't get or invalid branch hint value for hint "_s, j);

            BranchHint branchHint = static_cast<BranchHint>(parsedBranchHint);
            ASSERT(isValidBranchHint(branchHint));

            branchHintsForFunction.add(branchOffset, branchHint);
        }
        m_info->branchHints.add(functionIndex, WTFMove(branchHintsForFunction));
    }
    return { };
}

}
} // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
