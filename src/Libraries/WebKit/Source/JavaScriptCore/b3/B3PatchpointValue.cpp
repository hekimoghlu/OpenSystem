/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "B3PatchpointValue.h"

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

PatchpointValue::~PatchpointValue() = default;

void PatchpointValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    Base::dumpMeta(comma, out);
    out.print(comma, "resultConstraints = "_s);
    out.print(resultConstraints.size() > 1 ? "["_s : ""_s);

    CommaPrinter constraintComma;
    for (const auto& constraint : resultConstraints)
        out.print(constraintComma, constraint);
    out.print(resultConstraints.size() > 1 ? "]"_s : ""_s);

    if (numGPScratchRegisters)
        out.print(comma, "numGPScratchRegisters = "_s, numGPScratchRegisters);
    if (numFPScratchRegisters)
        out.print(comma, "numFPScratchRegisters = "_s, numFPScratchRegisters);
}

PatchpointValue::PatchpointValue(Type type, Origin origin, Kind kind)
    : Base(CheckedOpcode, kind, type, origin)
    , effects(Effects::forCall())
{
    ASSERT(accepts(kind));
    if (!type.isTuple())
        resultConstraints.append(type == Void ? ValueRep::WarmAny : ValueRep::SomeRegister);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
