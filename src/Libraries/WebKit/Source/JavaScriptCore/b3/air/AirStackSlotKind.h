/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 { namespace Air {

enum class StackSlotKind : uint8_t {
    // A locked stack slot is an area of stack requested by the client. It cannot be killed. The
    // client can get its FP offset and write to it from stack walking code, so we must assume
    // that reads and writes to a locked stack slot can be clobbered the same way as reads and
    // writes to any memory location.
    Locked,

    // A spill slot. These have fundamentally different behavior than a typical memory location.
    // They are lowered to from temporaries. This means for example that a 32-bit ZDef store to a
    // 8 byte stack slot will zero the top 4 bytes, even though a 32-bit ZDef store to any other
    // kind of memory location would do no such thing. UseAddr on a spill slot is not allowed, so
    // they never escape.
    Spill

    // FIXME: We should add a third mode, which means that the stack slot will be read asynchronously
    // as with Locked, but never written to asynchronously. Then, Air could optimize spilling and
    // filling by tracking whether the value had been stored to a read-only locked slot. If it had,
    // then we can refill from that slot.
    // https://bugs.webkit.org/show_bug.cgi?id=150587
};

} } } // namespace JSC::B3::Air

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::B3::Air::StackSlotKind);

} // namespace WTF

#endif // ENABLE(B3_JIT)
