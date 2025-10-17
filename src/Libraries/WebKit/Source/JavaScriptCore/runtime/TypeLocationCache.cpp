/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#include "TypeLocationCache.h"

#include "ControlFlowProfiler.h"
#include "TypeProfiler.h"
#include "VM.h"

namespace JSC {

std::pair<TypeLocation*, bool> TypeLocationCache::getTypeLocation(GlobalVariableID globalVariableID, SourceID sourceID, unsigned start, unsigned end, RefPtr<TypeSet>&& globalTypeSet, VM* vm)
{
    LocationKey key;
    key.m_globalVariableID = globalVariableID;
    key.m_sourceID = sourceID;
    key.m_start = start;
    key.m_end = end;

    auto result = m_locationMap.ensure(key, [&]{
        ASSERT(vm->typeProfiler());
        auto* location = vm->typeProfiler()->nextTypeLocation();
        location->m_globalVariableID = globalVariableID;
        location->m_sourceID = sourceID;
        location->m_divotStart = start;
        location->m_divotEnd = end;
        location->m_globalTypeSet = WTFMove(globalTypeSet);
        return location;
    });
    return std::pair { result.iterator->value, result.isNewEntry };
}

} // namespace JSC
