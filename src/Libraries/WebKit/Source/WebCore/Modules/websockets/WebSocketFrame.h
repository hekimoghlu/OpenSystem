/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

struct WebSocketFrame {
    // RFC6455 opcodes.
    enum OpCode {
        OpCodeContinuation = 0x0,
        OpCodeText = 0x1,
        OpCodeBinary = 0x2,
        OpCodeClose = 0x8,
        OpCodePing = 0x9,
        OpCodePong = 0xA,
        OpCodeInvalid = 0x10
    };

    enum ParseFrameResult {
        FrameOK,
        FrameIncomplete,
        FrameError
    };

    static bool isNonControlOpCode(OpCode opCode) { return opCode == OpCodeContinuation || opCode == OpCodeText || opCode == OpCodeBinary; }
    static bool isControlOpCode(OpCode opCode) { return opCode == OpCodeClose || opCode == OpCodePing || opCode == OpCodePong; }
    static bool isReservedOpCode(OpCode opCode) { return !isNonControlOpCode(opCode) && !isControlOpCode(opCode); }
    WEBCORE_EXPORT static bool needsExtendedLengthField(size_t payloadLength);
    WEBCORE_EXPORT static ParseFrameResult parseFrame(std::span<uint8_t> data, WebSocketFrame&, const uint8_t*& frameEnd, String& errorString); // May modify part of data to unmask the frame.

    WEBCORE_EXPORT WebSocketFrame(OpCode = OpCodeInvalid, bool final = false, bool compress = false, bool masked = false, std::span<const uint8_t> payload = { });
    WEBCORE_EXPORT void makeFrameData(Vector<uint8_t>& frameData);

    OpCode opCode;
    bool final;
    bool compress;
    bool reserved2;
    bool reserved3;
    bool masked;
    std::span<const uint8_t> payload;
};

} // namespace WebCore
