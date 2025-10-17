/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#include "WasmNameSectionParser.h"

#if ENABLE(WEBASSEMBLY)

#include "WasmNameSection.h"
#include "WasmTypeDefinitionInlines.h"

namespace JSC { namespace Wasm {

auto NameSectionParser::parse() -> Result
{
    Ref<NameSection> nameSection = NameSection::create();
    WASM_PARSER_FAIL_IF(!nameSection->functionNames.tryReserveCapacity(m_info.functionIndexSpaceSize()), "can't allocate enough memory for function names"_s);
    nameSection->functionNames.resize(m_info.functionIndexSpaceSize());

    for (size_t payloadNumber = 0; m_offset < source().size(); ++payloadNumber) {
        uint8_t nameType;
        uint32_t payloadLength;
        WASM_PARSER_FAIL_IF(!parseUInt7(nameType), "can't get name type for payload "_s, payloadNumber);
        WASM_PARSER_FAIL_IF(!parseVarUInt32(payloadLength), "can't get payload length for payload "_s, payloadNumber);
        WASM_PARSER_FAIL_IF(payloadLength > source().size() - m_offset, "payload length is too big for payload "_s, payloadNumber);
        const auto payloadStart = m_offset;

        if (!isValidNameType(nameType)) {
            // Unknown name section entries are simply ignored. This allows us to support newer toolchains without breaking older features.
            m_offset += payloadLength;
            continue;
        }

        switch (static_cast<NameType>(nameType)) {
        case NameType::Module: {
            uint32_t nameLen;
            Name nameString;
            WASM_PARSER_FAIL_IF(!parseVarUInt32(nameLen), "can't get module's name length for payload "_s, payloadNumber);
            WASM_PARSER_FAIL_IF(!consumeUTF8String(nameString, nameLen), "can't get module's name of length "_s, nameLen, " for payload "_s, payloadNumber);
            nameSection->moduleName = WTFMove(nameString);
            break;
        }
        case NameType::Function: {
            uint32_t count;
            WASM_PARSER_FAIL_IF(!parseVarUInt32(count), "can't get function count for payload "_s, payloadNumber);
            for (uint32_t function = 0; function < count; ++function) {
                uint32_t index;
                uint32_t nameLen;
                Name nameString;
                WASM_PARSER_FAIL_IF(!parseVarUInt32(index), "can't get function "_s, function, " index for payload "_s, payloadNumber);
                WASM_PARSER_FAIL_IF(m_info.functionIndexSpaceSize() <= index, "function "_s, function, " index "_s, index, " is larger than function index space "_s, m_info.functionIndexSpaceSize(), " for payload "_s, payloadNumber);
                WASM_PARSER_FAIL_IF(!parseVarUInt32(nameLen), "can't get functions "_s, function, "'s name length for payload "_s, payloadNumber);
                WASM_PARSER_FAIL_IF(!consumeUTF8String(nameString, nameLen), "can't get function "_s, function, "'s name of length "_s, nameLen, " for payload "_s, payloadNumber);
                nameSection->functionNames[index] = WTFMove(nameString);
            }
            break;
        }
        case NameType::Local: {
            // Ignore local names for now, we don't do anything with them but we still need to parse them in order to properly ignore them.
            uint32_t functionCount;
            WASM_PARSER_FAIL_IF(!parseVarUInt32(functionCount), "can't get function count for local name payload "_s, payloadNumber);
            for (uint32_t function = 0; function < functionCount; ++function) {
                uint32_t functionIndex;
                uint32_t count;
                WASM_PARSER_FAIL_IF(!parseVarUInt32(functionIndex), "can't get local's function index for payload "_s, payloadNumber);
                WASM_PARSER_FAIL_IF(!parseVarUInt32(count), "can't get local count for payload "_s, payloadNumber);
                for (uint32_t local = 0; local < count; ++local) {
                    uint32_t index;
                    uint32_t nameLen;
                    Name nameString;
                    WASM_PARSER_FAIL_IF(!parseVarUInt32(index), "can't get local "_s, local, " index for payload "_s, payloadNumber);
                    WASM_PARSER_FAIL_IF(!parseVarUInt32(nameLen), "can't get local "_s, local, "'s name length for payload "_s, payloadNumber);
                    WASM_PARSER_FAIL_IF(!consumeUTF8String(nameString, nameLen), "can't get local "_s, local, "'s name of length "_s, nameLen, " for payload "_s, payloadNumber);
                }
            }
            break;
        }
        }
        WASM_PARSER_FAIL_IF(payloadStart + payloadLength != m_offset, "payload for name section is not correct size, expected "_s, payloadLength, " got "_s, m_offset - payloadStart);
    }
    return nameSection;
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
