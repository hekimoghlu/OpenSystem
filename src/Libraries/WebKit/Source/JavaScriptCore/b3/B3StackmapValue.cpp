/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#include "B3StackmapValue.h"

#if ENABLE(B3_JIT)

#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

namespace B3StackmapValueInternal {
constexpr bool dumpRegisters = false;
}

StackmapValue::~StackmapValue() = default;

void StackmapValue::append(Value* value, const ValueRep& rep)
{
    if (rep == ValueRep::ColdAny) {
        childrenVector().append(value);
        return;
    }

    while (m_reps.size() < numChildren())
        m_reps.append(ValueRep::ColdAny);

    childrenVector().append(value);
    m_reps.append(rep);
}

void StackmapValue::appendSomeRegister(Value* value)
{
    append(ConstrainedValue(value, ValueRep::SomeRegister));
}

void StackmapValue::appendSomeRegisterWithClobber(Value* value)
{
    append(ConstrainedValue(value, ValueRep::SomeRegisterWithClobber));
}

void StackmapValue::setConstrainedChild(unsigned index, const ConstrainedValue& constrainedValue)
{
    child(index) = constrainedValue.value();
    setConstraint(index, constrainedValue.rep());
}

void StackmapValue::setConstraint(unsigned index, const ValueRep& rep)
{
    if (rep == ValueRep(ValueRep::ColdAny))
        return;

    while (m_reps.size() <= index)
        m_reps.append(ValueRep::ColdAny);

    m_reps[index] = rep;
}

void StackmapValue::dumpChildren(CommaPrinter& comma, PrintStream& out) const
{
    for (ConstrainedValue value : constrainedChildren())
        out.print(comma, value);
}

void StackmapValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(
        comma, "generator = ", RawPointer(m_generator.get()));
    if constexpr (B3StackmapValueInternal::dumpRegisters) {
        out.print(", earlyClobbered = ", m_earlyClobbered,
            ", lateClobbered = ", m_lateClobbered, ", usedRegisters = ", m_usedRegisters);
    }
}

StackmapValue::StackmapValue(CheckedOpcodeTag, Kind kind, Type type, Origin origin)
    : Value(CheckedOpcode, kind, type, VarArgs, origin)
{
    ASSERT(accepts(kind));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

