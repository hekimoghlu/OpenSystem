/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#include "WasmFunctionIPIntMetadataGenerator.h"

#include <numeric>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEBASSEMBLY)

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

namespace Wasm {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FunctionIPIntMetadataGenerator);

unsigned FunctionIPIntMetadataGenerator::addSignature(const TypeDefinition& signature)
{
    unsigned index = m_signatures.size();
    m_signatures.append(&signature);
    return index;
}

void FunctionIPIntMetadataGenerator::setTailCall(uint32_t functionIndex, bool isImportedFunctionFromFunctionIndexSpace)
{
    m_tailCallSuccessors.set(functionIndex);
    if (isImportedFunctionFromFunctionIndexSpace)
        setTailCallClobbersInstance();
}

void FunctionIPIntMetadataGenerator::addLength(size_t length)
{
    IPInt::InstructionLengthMetadata instructionLength {
        .length = safeCast<uint8_t>(length)
    };
    size_t size = m_metadata.size();
    m_metadata.grow(size + sizeof(instructionLength));
    WRITE_TO_METADATA(m_metadata.data() + size, instructionLength, IPInt::InstructionLengthMetadata);
}

void FunctionIPIntMetadataGenerator::addLEB128ConstantInt32AndLength(uint32_t value, size_t length)
{
    IPInt::Const32Metadata mdConst {
        .instructionLength = { .length = safeCast<uint8_t>(length) },
        .value = value
    };
    size_t size = m_metadata.size();
    m_metadata.grow(size + sizeof(mdConst));
    WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::Const32Metadata);
}

void FunctionIPIntMetadataGenerator::addLEB128ConstantAndLengthForType(Type type, uint64_t value, size_t length)
{
    if (type.isI32()) {
        size_t size = m_metadata.size();
        if (length == 2) {
            IPInt::InstructionLengthMetadata mdConst {
                .length = safeCast<uint8_t>((value >> 7) & 1)
            };
            m_metadata.grow(size + sizeof(mdConst));
            WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::InstructionLengthMetadata);
        } else {
            IPInt::Const32Metadata mdConst {
                .instructionLength = { .length = safeCast<uint8_t>(length) },
                .value = static_cast<uint32_t>(value)
            };
            m_metadata.grow(size + sizeof(mdConst));
            WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::Const32Metadata);
        }
    } else if (type.isI64()) {
        size_t size = m_metadata.size();
        IPInt::Const64Metadata mdConst {
            .instructionLength = { .length = safeCast<uint8_t>(length) },
            .value = static_cast<uint64_t>(value)
        };
        m_metadata.grow(size + sizeof(mdConst));
        WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::Const64Metadata);
    } else if (type.isRef() || type.isRefNull() || type.isFuncref()) {
        size_t size = m_metadata.size();
        IPInt::Const32Metadata mdConst {
            .instructionLength = { .length = safeCast<uint8_t>(length) },
            .value = static_cast<uint32_t>(value)
        };
        m_metadata.grow(size + sizeof(mdConst));
        WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::Const32Metadata);
    } else if (!type.isF32() && !type.isF64())
        ASSERT_NOT_IMPLEMENTED_YET();
}

void FunctionIPIntMetadataGenerator::addLEB128V128Constant(v128_t value, size_t length)
{
    IPInt::Const128Metadata mdConst {
        .instructionLength = { .length = safeCast<uint8_t>(length) },
        .value = value
    };
    size_t size = m_metadata.size();
    m_metadata.grow(size + sizeof(mdConst));
    WRITE_TO_METADATA(m_metadata.data() + size, mdConst, IPInt::Const128Metadata);
}

void FunctionIPIntMetadataGenerator::addReturnData(const FunctionSignature& sig)
{
    CallInformation returnCC = wasmCallingConvention().callInformationFor(sig, CallRole::Callee);
    m_uINTBytecode.reserveInitialCapacity(sig.returnCount() + 1);
    // uINT: the interpreter smaller than mINT
    // 0x00-0x07: r0 - r7
    // 0x08-0x0f: f0 - f7
    // 0x10: stack
    // 0x11: return

    constexpr static int NUM_UINT_GPRS = 8;
    constexpr static int NUM_UINT_FPRS = 8;
    ASSERT_UNUSED(NUM_UINT_GPRS, wasmCallingConvention().jsrArgs.size() <= NUM_UINT_GPRS);
    ASSERT_UNUSED(NUM_UINT_FPRS, wasmCallingConvention().fprArgs.size() <= NUM_UINT_FPRS);

    for (size_t i = 0; i < sig.returnCount(); ++i) {
        auto loc = returnCC.results[i].location;

        if (loc.isGPR()) {
#if USE(JSVALUE64)
            ASSERT_UNUSED(NUM_UINT_GPRS, GPRInfo::toArgumentIndex(loc.jsr().gpr()) < NUM_UINT_GPRS);
            m_uINTBytecode.append(static_cast<uint8_t>(IPInt::UIntBytecode::RetGPR) + GPRInfo::toArgumentIndex(loc.jsr().gpr()));
#elif USE(JSVALUE32_64)
            ASSERT_UNUSED(NUM_UINT_GPRS, GPRInfo::toArgumentIndex(loc.jsr().payloadGPR()) < NUM_UINT_GPRS);
            ASSERT_UNUSED(NUM_UINT_GPRS, GPRInfo::toArgumentIndex(loc.jsr().tagGPR()) < NUM_UINT_GPRS);
            m_uINTBytecode.append(static_cast<uint8_t>(IPInt::UIntBytecode::RetGPR) + GPRInfo::toArgumentIndex(loc.jsr().gpr(WhichValueWord::PayloadWord)));
#endif
        } else if (loc.isFPR()) {
            ASSERT_UNUSED(NUM_UINT_FPRS, FPRInfo::toArgumentIndex(loc.fpr()) < NUM_UINT_FPRS);
            m_uINTBytecode.append(static_cast<uint8_t>(IPInt::UIntBytecode::RetFPR) + FPRInfo::toArgumentIndex(loc.fpr()));
        } else if (loc.isStack()) {
            m_highestReturnStackOffset = loc.offsetFromFP();
            m_uINTBytecode.append(static_cast<uint8_t>(IPInt::UIntBytecode::Stack));
        }
    }
    m_uINTBytecode.reverse();
    m_uINTBytecode.append(static_cast<uint8_t>(IPInt::UIntBytecode::End));
}

} }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
