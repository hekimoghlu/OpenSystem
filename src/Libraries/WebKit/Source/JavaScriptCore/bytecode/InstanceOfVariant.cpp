/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#include "InstanceOfVariant.h"

#include "JSCJSValueInlines.h"

namespace JSC {

InstanceOfVariant::InstanceOfVariant(
    const StructureSet& structureSet, const ObjectPropertyConditionSet& conditionSet,
    JSObject* prototype, bool isHit)
    : m_structureSet(structureSet)
    , m_conditionSet(conditionSet)
    , m_prototype(prototype)
    , m_isHit(isHit)
{
}

bool InstanceOfVariant::attemptToMerge(const InstanceOfVariant& other)
{
    if (m_prototype != other.m_prototype)
        return false;
    
    if (m_isHit != other.m_isHit)
        return false;
    
    ObjectPropertyConditionSet mergedConditionSet = m_conditionSet.mergedWith(other.m_conditionSet);
    if (!mergedConditionSet.isValid())
        return false;
    m_conditionSet = mergedConditionSet;
    
    m_structureSet.merge(other.m_structureSet);
    
    return true;
}

void InstanceOfVariant::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void InstanceOfVariant::dumpInContext(PrintStream& out, DumpContext* context) const
{
    if (!*this) {
        out.print("<empty>");
        return;
    }
    
    out.print(
        "<", inContext(structureSet(), context), ", ", inContext(m_conditionSet, context), ","
        " prototype = ", inContext(JSValue(m_prototype), context), ","
        " isHit = ", m_isHit, ">");
}

} // namespace JSC

