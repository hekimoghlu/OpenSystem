/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

#include "WasmFormat.h"

#if ENABLE(WEBASSEMBLY)

#include <wtf/CheckedArithmetic.h>
#include <wtf/FastMalloc.h>
#include <wtf/text/MakeString.h>

namespace JSC { namespace Wasm {

constexpr uintptr_t NullWasmCallee = 0;

Segment::Ptr Segment::create(std::optional<I32InitExpr> offset, uint32_t sizeInBytes, Kind kind)
{
    CheckedUint32 totalBytesChecked = sizeInBytes;
    totalBytesChecked += sizeof(Segment);
    if (totalBytesChecked.hasOverflowed())
        return Ptr(nullptr, &Segment::destroy);
    auto allocated = tryFastCalloc(totalBytesChecked, 1);
    Segment* segment;
    if (!allocated.getValue(segment))
        return Ptr(nullptr, &Segment::destroy);
    ASSERT(kind == Kind::Passive || !!offset);
    segment->kind = kind;
    segment->offsetIfActive = WTFMove(offset);
    segment->sizeInBytes = sizeInBytes;
    return Ptr(segment, &Segment::destroy);
}

void Segment::destroy(Segment *segment)
{
    fastFree(segment);
}

String makeString(const Name& characters)
{
    return WTF::makeString(characters);
}

} } // namespace JSC::Wasm

namespace WTF {

void printInternal(PrintStream& out, JSC::Wasm::TableElementType type)
{
    switch (type) {
    case JSC::Wasm::TableElementType::Externref:
        out.print("Externref");
        break;
    case JSC::Wasm::TableElementType::Funcref:
        out.print("Funcref");
        break;
    }
}

} // namespace WTF

#endif // ENABLE(WEBASSEMBLY)
