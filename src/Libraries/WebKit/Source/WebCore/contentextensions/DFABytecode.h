/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore::ContentExtensions {

using DFABytecode = uint8_t;

// Increment ContentExtensionStore::CurrentContentExtensionFileVersion
// when making any non-backwards-compatible changes to the bytecode.
// FIXME: Changes here should not require changes in WebKit2.  Move all versioning to WebCore.
enum class DFABytecodeInstruction : uint8_t {

    // CheckValue has two arguments:
    // The value to check (1 byte),
    // The distance to jump if the values are equal (1-4 bytes, signed).
    CheckValueCaseInsensitive = 0x0,
    CheckValueCaseSensitive = 0x1,

    // Jump table if the input value is within a certain range.
    // The lower value (1 byte).
    // The higher value (1 byte).
    // The distance to jump if the value is in the range
    // for every character in the range (1-4 bytes, signed).
    JumpTableCaseInsensitive = 0x2,
    JumpTableCaseSensitive = 0x3,

    // Jump to an offset if the input value is within a certain range.
    // The lower value (1 byte).
    // The higher value (1 byte).
    // The distance to jump if the value is in the range (1-4 bytes, signed).
    CheckValueRangeCaseInsensitive = 0x4,
    CheckValueRangeCaseSensitive = 0x5,

    // AppendAction has one argument:
    // The action to append (1-4 bytes).
    AppendAction = 0x6,
    
    // TestFlagsAndAppendAction has two arguments:
    // The flags to check before appending (1-3 bytes).
    // The action to append (1-4 bytes).
    TestFlagsAndAppendAction = 0x8,

    // Terminate has no arguments.
    Terminate = 0xA,

    // Jump has one argument:
    // The distance to jump (1-4 bytes, signed).
    Jump = 0xB,
};

// The last four bits contain the instruction type.
static constexpr uint8_t DFABytecodeInstructionMask = 0x0F;
static constexpr uint8_t DFABytecodeJumpSizeMask = 0x30;
static constexpr uint8_t DFABytecodeFlagsSizeMask = 0x30;
static constexpr uint8_t DFABytecodeActionSizeMask = 0xC0;

// DFA bytecode starts with a 4 byte header which contains the size of this DFA.
using DFAHeader = uint32_t;

// DFABytecodeFlagsSize and DFABytecodeActionSize are stored in the top four bits of the DFABytecodeInstructions that have flags and actions.
enum class DFABytecodeFlagsSize : uint8_t {
    UInt8 = 0x00,
    UInt16 = 0x10,
    UInt24 = 0x20,
};
enum class DFABytecodeActionSize : uint8_t {
    UInt8 = 0x00,
    UInt16 = 0x40,
    UInt24 = 0x80,
    UInt32 = 0xC0,
};

// A DFABytecodeJumpSize is stored in the top four bits of the DFABytecodeInstructions that have a jump.
enum class DFABytecodeJumpSize : uint8_t {
    Int8 = 0x00,
    Int16 = 0x10,
    Int24 = 0x20,
    Int32 = 0x30,
};
static constexpr int32_t UInt24Max = (1 << 24) - 1;
static constexpr int32_t Int24Max = (1 << 23) - 1;
static constexpr int32_t Int24Min = -(1 << 23);
static constexpr size_t Int24Size = 3;
static constexpr size_t UInt24Size = 3;

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
