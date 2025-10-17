/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#include "ComplexGetStatus.h"

#include "StructureInlines.h"

namespace JSC {

ComplexGetStatus ComplexGetStatus::computeFor(
    Structure* headStructure, const ObjectPropertyConditionSet& conditionSet, UniquedStringImpl* uid)
{
    // FIXME: We should assert that we never see a structure that
    // getOwnPropertySlotIsImpure() but for which we don't
    // newImpurePropertyFiresWatchpoints(). We're not at a point where we can do
    // that, yet.
    // https://bugs.webkit.org/show_bug.cgi?id=131810
    
    ASSERT(conditionSet.isValid());
    
    if (headStructure->takesSlowPathInDFGForImpureProperty())
        return takesSlowPath();
    
    ComplexGetStatus result;
    result.m_kind = Inlineable;
    
    if (!conditionSet.isEmpty()) {
        result.m_conditionSet = conditionSet;
        
        if (!result.m_conditionSet.structuresEnsureValidity())
            return skip();

        unsigned numberOfSlotBases =
            result.m_conditionSet.numberOfConditionsWithKind(PropertyCondition::Presence);
        RELEASE_ASSERT(numberOfSlotBases <= 1);
        if (!numberOfSlotBases) {
            ASSERT(result.m_offset == invalidOffset);
            return result;
        }
        ObjectPropertyCondition base = result.m_conditionSet.slotBaseCondition();
        ASSERT(base.kind() == PropertyCondition::Presence);

        result.m_offset = base.offset();
    } else
        result.m_offset = headStructure->getConcurrently(uid);
    
    if (!isValidOffset(result.m_offset))
        return takesSlowPath();
    
    return result;
}

} // namespace JSC


