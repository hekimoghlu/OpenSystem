/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "TypeProfiler.h"

#include "TypeLocation.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace JSC {

namespace TypeProfilerInternal {
static constexpr bool verbose = false;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(TypeProfiler);

TypeProfiler::TypeProfiler()
    : m_nextUniqueVariableID(1)
{ 
}

void TypeProfiler::logTypesForTypeLocation(TypeLocation* location, VM& vm)
{
    TypeProfilerSearchDescriptor descriptor = location->m_globalVariableID == TypeProfilerReturnStatement ? TypeProfilerSearchDescriptorFunctionReturn : TypeProfilerSearchDescriptorNormal;

    dataLogF("[Start, End]::[%u, %u]\n", location->m_divotStart, location->m_divotEnd);

    if (findLocation(location->m_divotStart, location->m_sourceID, descriptor, vm))
        dataLog("\t\t[Entry IS in System]\n");
    else
        dataLog("\t\t[Entry IS NOT in system]\n");

    dataLog("\t\t", location->m_globalVariableID == TypeProfilerReturnStatement ? "[Return Statement]" : "[Normal Statement]", "\n");

    dataLog("\t\t#Local#\n\t\t", makeStringByReplacingAll(location->m_instructionTypeSet->dumpTypes(), '\n', "\n\t\t"_s), "\n");
    if (location->m_globalTypeSet)
        dataLog("\t\t#Global#\n\t\t", makeStringByReplacingAll(location->m_globalTypeSet->dumpTypes(), '\n', "\n\t\t"_s), "\n");
}

void TypeProfiler::insertNewLocation(TypeLocation* location)
{
    if (TypeProfilerInternal::verbose)
        dataLogF("Registering location:: divotStart:%u, divotEnd:%u\n", location->m_divotStart, location->m_divotEnd);

    if (!m_bucketMap.contains(location->m_sourceID)) {
        Vector<TypeLocation*> bucket;
        m_bucketMap.set(location->m_sourceID, bucket);
    }

    Vector<TypeLocation*>& bucket = m_bucketMap.find(location->m_sourceID)->value;
    bucket.append(location);
}

String TypeProfiler::typeInformationForExpressionAtOffset(TypeProfilerSearchDescriptor descriptor, unsigned offset, SourceID sourceID, VM& vm)
{
    // This returns a JSON string representing an Object with the following properties:
    //     globalTypeSet: 'JSON<TypeSet> | null'
    //     instructionTypeSet: 'JSON<TypeSet>'

    TypeLocation* location = findLocation(offset, sourceID, descriptor, vm);
    ASSERT(location);

    StringBuilder json;  

    json.append('{');

    json.append("\"globalTypeSet\":"_s);
    if (location->m_globalTypeSet && location->m_globalVariableID != TypeProfilerNoGlobalIDExists)
        json.append(location->m_globalTypeSet->toJSONString());
    else
        json.append("null"_s);
    json.append(',');

    json.append("\"instructionTypeSet\":"_s, location->m_instructionTypeSet->toJSONString(), ',');

    bool isOverflown = location->m_instructionTypeSet->isOverflown() || (location->m_globalTypeSet && location->m_globalTypeSet->isOverflown());
    json.append("\"isOverflown\":"_s, isOverflown ? "true"_s : "false"_s);

    json.append('}');

    return json.toString();
}

TypeLocation* TypeProfiler::findLocation(unsigned divot, SourceID sourceID, TypeProfilerSearchDescriptor descriptor, VM& vm)
{
    QueryKey queryKey(sourceID, divot, descriptor);
    auto iter = m_queryCache.find(queryKey);
    if (iter != m_queryCache.end())
        return iter->value;

    if (!vm.functionHasExecutedCache()->hasExecutedAtOffset(sourceID, divot))
        return nullptr;

    if (!m_bucketMap.contains(sourceID))
        return nullptr;

    Vector<TypeLocation*>& bucket = m_bucketMap.find(sourceID)->value;
    TypeLocation* bestMatch = nullptr;
    unsigned distance = UINT_MAX; // Because assignments may be nested, make sure we find the closest enclosing assignment to this character offset.
    for (auto* location : bucket) {
        // We found the type location that correlates to the convergence of all return statements in a function.
        // This text offset is the offset of the opening brace in a function declaration.
        if (descriptor == TypeProfilerSearchDescriptorFunctionReturn && location->m_globalVariableID == TypeProfilerReturnStatement && location->m_divotForFunctionOffsetIfReturnStatement == divot)
            return location;

        if (descriptor != TypeProfilerSearchDescriptorFunctionReturn && location->m_globalVariableID != TypeProfilerReturnStatement && location->m_divotStart <= divot && divot <= location->m_divotEnd && location->m_divotEnd - location->m_divotStart <= distance) {
            distance = location->m_divotEnd - location->m_divotStart;
            bestMatch = location;
        }
    }

    if (bestMatch)
        m_queryCache.set(queryKey, bestMatch);
    // FIXME: BestMatch should never be null past this point. This doesn't hold currently because we ignore var assignments when code contains eval/With (VarInjection). 
    // https://bugs.webkit.org/show_bug.cgi?id=135184
    return bestMatch;
}

TypeLocation* TypeProfiler::nextTypeLocation() 
{ 
    return m_typeLocationInfo.add(); 
}

void TypeProfiler::invalidateTypeSetCache(VM& vm)
{
    for (Bag<TypeLocation>::iterator iter = m_typeLocationInfo.begin(); !!iter; ++iter) {
        TypeLocation* location = *iter;
        location->m_instructionTypeSet->invalidateCache(vm);
        if (location->m_globalTypeSet)
            location->m_globalTypeSet->invalidateCache(vm);
    }
}

void TypeProfiler::dumpTypeProfilerData(VM& vm)
{
    for (Bag<TypeLocation>::iterator iter = m_typeLocationInfo.begin(); !!iter; ++iter) {
        TypeLocation* location = *iter;
        logTypesForTypeLocation(location, vm);
    }
}

} // namespace JSC
