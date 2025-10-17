/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include "InstanceOfAccessCase.h"

#if ENABLE(JIT)

#include "JSCJSValueInlines.h"

namespace JSC {

Ref<AccessCase> InstanceOfAccessCase::create(
    VM& vm, JSCell* owner, AccessType accessType, Structure* structure,
    const ObjectPropertyConditionSet& conditionSet, JSObject* prototype)
{
    ASSERT(accessType == AccessCase::InstanceOfMiss || accessType == AccessCase::InstanceOfHit);
    return adoptRef(*new InstanceOfAccessCase(vm, owner, accessType, structure, conditionSet, prototype));
}

void InstanceOfAccessCase::dumpImpl(PrintStream& out, CommaPrinter& comma, Indenter& indent) const
{
    Base::dumpImpl(out, comma, indent);
    out.print(comma, "prototype = ", JSValue(prototype()));
}

InstanceOfAccessCase::InstanceOfAccessCase(
    VM& vm, JSCell* owner, AccessType accessType, Structure* structure,
    const ObjectPropertyConditionSet& conditionSet, JSObject* prototype)
    : Base(vm, owner, accessType, nullptr, invalidOffset, structure, conditionSet, nullptr)
    , m_prototype(vm, owner, prototype)
{
}

} // namespace JSC

#endif // ENABLE(JIT)

