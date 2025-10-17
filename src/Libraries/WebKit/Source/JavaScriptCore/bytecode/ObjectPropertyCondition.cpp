/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#include "ObjectPropertyCondition.h"

#include "JSCJSValueInlines.h"
#include "TrackedReferences.h"

namespace JSC {

void ObjectPropertyCondition::dumpInContext(PrintStream& out, DumpContext* context) const
{
    if (!*this) {
        out.print("<invalid>");
        return;
    }
    
    out.print("<", inContext(JSValue(m_object), context), ": ", inContext(m_condition, context), ">");
}

void ObjectPropertyCondition::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

bool ObjectPropertyCondition::structureEnsuresValidityAssumingImpurePropertyWatchpoint(Concurrency concurrency) const
{
    if (!*this)
        return false;
    
    return m_condition.isStillValidAssumingImpurePropertyWatchpoint(concurrency, m_object->structure(), nullptr);
}

bool ObjectPropertyCondition::validityRequiresImpurePropertyWatchpoint(Structure* structure) const
{
    return m_condition.validityRequiresImpurePropertyWatchpoint(structure);
}

bool ObjectPropertyCondition::validityRequiresImpurePropertyWatchpoint() const
{
    if (!*this)
        return false;
    
    return validityRequiresImpurePropertyWatchpoint(m_object->structure());
}

bool ObjectPropertyCondition::isStillValidAssumingImpurePropertyWatchpoint(Concurrency concurrency, Structure* structure) const
{
    return m_condition.isStillValidAssumingImpurePropertyWatchpoint(concurrency, structure, m_object);
}

bool ObjectPropertyCondition::isStillValidAssumingImpurePropertyWatchpoint(Concurrency concurrency) const
{
    if (!*this)
        return false;

    return isStillValidAssumingImpurePropertyWatchpoint(concurrency, m_object->structure());
}


bool ObjectPropertyCondition::isStillValid(Concurrency concurrency, Structure* structure) const
{
    return m_condition.isStillValid(concurrency, structure, m_object);
}

bool ObjectPropertyCondition::isStillValid(Concurrency concurrency) const
{
    if (!*this)
        return false;
    
    return isStillValid(concurrency, m_object->structure());
}

bool ObjectPropertyCondition::structureEnsuresValidity(Concurrency concurrency, Structure* structure) const
{
    return m_condition.isStillValid(concurrency, structure);
}

bool ObjectPropertyCondition::structureEnsuresValidity(Concurrency concurrency) const
{
    if (!*this)
        return false;
    
    return structureEnsuresValidity(concurrency, m_object->structure());
}

bool ObjectPropertyCondition::isWatchableAssumingImpurePropertyWatchpoint(Structure* structure, PropertyCondition::WatchabilityEffort effort, Concurrency concurrency) const
{
    return m_condition.isWatchableAssumingImpurePropertyWatchpoint(structure, m_object, effort, concurrency);
}

bool ObjectPropertyCondition::isWatchableAssumingImpurePropertyWatchpoint(Structure* structure, PropertyCondition::WatchabilityEffort effort) const
{
    return m_condition.isWatchableAssumingImpurePropertyWatchpoint(structure, m_object, effort);
}

bool ObjectPropertyCondition::isWatchableAssumingImpurePropertyWatchpoint(PropertyCondition::WatchabilityEffort effort, Concurrency concurrency) const
{
    if (!*this)
        return false;

    return isWatchableAssumingImpurePropertyWatchpoint(m_object->structure(), effort, concurrency);
}

bool ObjectPropertyCondition::isWatchableAssumingImpurePropertyWatchpoint(PropertyCondition::WatchabilityEffort effort) const
{
    if (!*this)
        return false;
    
    return isWatchableAssumingImpurePropertyWatchpoint(m_object->structure(), effort);
}

bool ObjectPropertyCondition::isWatchable(Structure* structure, PropertyCondition::WatchabilityEffort effort) const
{
    return m_condition.isWatchable(structure, m_object, effort);
}

bool ObjectPropertyCondition::isWatchable(PropertyCondition::WatchabilityEffort effort) const
{
    if (!*this)
        return false;
    return isWatchable(m_object->structure(), effort);
}

bool ObjectPropertyCondition::isWatchable(PropertyCondition::WatchabilityEffort effort, Concurrency concurrency) const
{
    if (!*this)
        return false;
    return m_condition.isWatchable(m_object->structure(), m_object, effort, concurrency);
}

bool ObjectPropertyCondition::isStillLive(VM& vm) const
{
    if (!*this)
        return false;
    
    bool isStillLive = true;
    forEachDependentCell([&](JSCell* cell) {
        isStillLive &= vm.heap.isMarked(cell);
    });
    return isStillLive;
}

void ObjectPropertyCondition::validateReferences(const TrackedReferences& tracked) const
{
    if (!*this)
        return;
    
    tracked.check(m_object);
    m_condition.validateReferences(tracked);
}

ObjectPropertyCondition ObjectPropertyCondition::attemptToMakeEquivalenceWithoutBarrier() const
{
    PropertyCondition result = condition().attemptToMakeEquivalenceWithoutBarrier(object());
    if (!result)
        return ObjectPropertyCondition();
    return ObjectPropertyCondition(object(), result);
}

ObjectPropertyCondition ObjectPropertyCondition::attemptToMakeReplacementWithoutBarrier() const
{
    PropertyCondition result = condition().attemptToMakeReplacementWithoutBarrier(object());
    if (!result)
        return ObjectPropertyCondition();
    return ObjectPropertyCondition(object(), result);
}

} // namespace JSC

