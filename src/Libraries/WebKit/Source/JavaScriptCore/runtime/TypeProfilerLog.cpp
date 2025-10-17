/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#include "TypeProfilerLog.h"

#include "FrameTracers.h"
#include "JSCJSValueInlines.h"
#include "TypeLocation.h"
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

namespace TypeProfilerLogInternal {
static constexpr bool verbose = false;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(TypeProfilerLog);

TypeProfilerLog::TypeProfilerLog(VM& vm)
    : m_vm(vm)
    , m_logSize(50000)
    , m_logStartPtr(new LogEntry[m_logSize])
    , m_currentLogEntryPtr(m_logStartPtr)
    , m_logEndPtr(m_logStartPtr + m_logSize)
{
    ASSERT(m_logStartPtr);
}

TypeProfilerLog::~TypeProfilerLog()
{
    delete[] m_logStartPtr;
}

void TypeProfilerLog::processLogEntries(VM& vm, const String& reason)
{
    // We need to do this because this code will call into calculatedDisplayName.
    // calculatedDisplayName will clear any exception it sees (because it thinks
    // it's a stack overflow). We may be called when an exception was already
    // thrown, so we don't want calculatedDisplayName to clear that exception that
    // was thrown before we even got here.
    SuspendExceptionScope suspendExceptionScope(vm);

    MonotonicTime before { };
    if (TypeProfilerLogInternal::verbose) {
        dataLog("Process caller:'", reason, "'");
        before = MonotonicTime::now();
    }

    UncheckedKeyHashMap<Structure*, RefPtr<StructureShape>> cachedMonoProtoShapes;
    UncheckedKeyHashMap<std::pair<Structure*, JSCell*>, RefPtr<StructureShape>> cachedPolyProtoShapes;

    LogEntry* entry = m_logStartPtr;

    while (entry != m_currentLogEntryPtr) {
        StructureID id = entry->structureID;
        RefPtr<StructureShape> shape;
        JSValue value = entry->value;
        Structure* structure = nullptr;
        bool sawPolyProtoStructure = false;
        if (id) {
            structure = id.decode();
            auto iter = cachedMonoProtoShapes.find(structure);
            if (iter == cachedMonoProtoShapes.end()) {
                auto key = std::make_pair(structure, value.asCell());
                auto iter = cachedPolyProtoShapes.find(key);
                if (iter != cachedPolyProtoShapes.end()) {
                    shape = iter->value;
                    sawPolyProtoStructure = true;
                }

                if (!shape) {
                    shape = structure->toStructureShape(value, sawPolyProtoStructure);
                    if (sawPolyProtoStructure)
                        cachedPolyProtoShapes.set(key, shape);
                    else
                        cachedMonoProtoShapes.set(structure, shape);
                }
            } else
                shape = iter->value;
        }

        RuntimeType type = runtimeTypeForValue(value);
        TypeLocation* location = entry->location;
        location->m_lastSeenType = type;
        if (location->m_globalTypeSet)
            location->m_globalTypeSet->addTypeInformation(type, shape.copyRef(), structure, sawPolyProtoStructure);
        location->m_instructionTypeSet->addTypeInformation(type, WTFMove(shape), structure, sawPolyProtoStructure);

        entry++;
    }

    // Note that we don't update this cursor until we're done processing the log.
    // This allows us to have a sane story in case we have to mark the log
    // while processing through it. We won't be iterating over the log while
    // marking it, but we may be in the middle of iterating over when the mutator
    // pauses and causes the collector to mark the log.
    m_currentLogEntryPtr = m_logStartPtr;

    if (TypeProfilerLogInternal::verbose) {
        MonotonicTime after = MonotonicTime::now();
        dataLogF(" Processing the log took: '%f' ms\n", (after - before).milliseconds());
    }
}

// We don't need a SlotVisitor version of this because TypeProfilerLog is only used by
// dev tools, and is therefore not on the critical path for performance.
void TypeProfilerLog::visit(AbstractSlotVisitor& visitor)
{
    for (LogEntry* entry = m_logStartPtr; entry != m_currentLogEntryPtr; ++entry) {
        visitor.appendUnbarriered(entry->value);
        if (StructureID id = entry->structureID) {
            Structure* structure = id.decode();
            visitor.appendUnbarriered(structure);
        }
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
