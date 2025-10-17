/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#pragma once

#include "JSCJSValue.h"
#include "Structure.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class AbstractSlotVisitor;
class TypeLocation;

class TypeProfilerLog {
    WTF_MAKE_TZONE_ALLOCATED(TypeProfilerLog);
public:
    struct LogEntry {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
    public:
        friend class LLIntOffsetsExtractor;

        JSValue value;
        TypeLocation* location; 
        StructureID structureID;

        static constexpr ptrdiff_t structureIDOffset() { return OBJECT_OFFSETOF(LogEntry, structureID); }
        static constexpr ptrdiff_t valueOffset() { return OBJECT_OFFSETOF(LogEntry, value); }
        static constexpr ptrdiff_t locationOffset() { return OBJECT_OFFSETOF(LogEntry, location); }
    };


    TypeProfilerLog(VM&);
    ~TypeProfilerLog();

    JS_EXPORT_PRIVATE void processLogEntries(VM&, const String&);
    LogEntry* logEndPtr() const { return m_logEndPtr; }

    void visit(AbstractSlotVisitor&);

    static constexpr ptrdiff_t logStartOffset() { return OBJECT_OFFSETOF(TypeProfilerLog, m_logStartPtr); }
    static constexpr ptrdiff_t currentLogEntryOffset() { return OBJECT_OFFSETOF(TypeProfilerLog, m_currentLogEntryPtr); }

private:
    friend class LLIntOffsetsExtractor;

    VM& m_vm;
    unsigned m_logSize;
    LogEntry* m_logStartPtr;
    LogEntry* m_currentLogEntryPtr;
    LogEntry* m_logEndPtr;
};

} // namespace JSC
