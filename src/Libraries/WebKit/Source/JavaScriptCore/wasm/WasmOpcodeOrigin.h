/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

#if ENABLE(WEBASSEMBLY_OMGJIT)

#include "B3Origin.h"
#include "WasmFormat.h"

#include <wtf/ForbidHeapAllocation.h>

namespace JSC { namespace Wasm {

class OpcodeOrigin {
    WTF_FORBID_HEAP_ALLOCATION;

public:
    void dump(PrintStream&) const;

    OpcodeOrigin() = default;

    friend bool operator==(const OpcodeOrigin&, const OpcodeOrigin&) = default;

#if USE(JSVALUE64)
    OpcodeOrigin(OpType opcode, size_t offset)
    {
        ASSERT(static_cast<uint32_t>(offset) == offset);
        ASSERT(static_cast<OpType>(static_cast<uint8_t>(opcode)) == opcode);
        packedData = (static_cast<uint64_t>(opcode) << 32) | offset;
    }
    OpcodeOrigin(OpType prefix, uint32_t opcode, size_t offset)
    {
        ASSERT(static_cast<uint32_t>(offset) == offset);
        ASSERT(static_cast<OpType>(static_cast<uint8_t>(prefix)) == prefix);
        ASSERT((opcode & ((1 << 24) - 1)) == opcode);
        packedData = (static_cast<uint64_t>(opcode) << 40) | (static_cast<uint64_t>(prefix) << 32) | offset;
    }
    OpcodeOrigin(B3::Origin origin)
        : packedData(std::bit_cast<uint64_t>(origin))
    {
    }

    OpType opcode() const { return static_cast<OpType>(packedData >> 32 & 0xff); }
    Ext1OpType ext1Opcode() const { return static_cast<Ext1OpType>(packedData >> 40); }
    ExtSIMDOpType simdOpcode() const { return static_cast<ExtSIMDOpType>(packedData >> 40); }
    ExtGCOpType gcOpcode() const { return static_cast<ExtGCOpType>(packedData >> 40); }
    ExtAtomicOpType atomicOpcode() const { return static_cast<ExtAtomicOpType>(packedData >> 40); }
    size_t location() const { return static_cast<uint32_t>(packedData); }

private:
    static_assert(sizeof(void*) == sizeof(uint64_t), "this packing doesn't work if this isn't the case");
    uint64_t packedData { 0 };

#elif USE(JSVALUE32_64)
    OpcodeOrigin(OpType prefix, size_t offset)
    {
        // We accept the wrap around for large offsets.
        packedData = (static_cast<uint32_t>(prefix) << 24) | (offset & 0xffffff);
    }

    OpcodeOrigin(OpType prefix, size_t, size_t offset)
    {
        // We accept the wrap around for large offsets.
        packedData = (static_cast<uint32_t>(prefix) << 24) | (offset & 0xffffff);
    }

    OpcodeOrigin(B3::Origin origin)
        : packedData(std::bit_cast<uint32_t>(origin))
    {
    }

    OpType opcode() const { return static_cast<OpType>(packedData >> 24); }
    size_t location() const { return packedData & 0xffffff; }
private:
    uint32_t packedData { 0 };
#endif
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)
