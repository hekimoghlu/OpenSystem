/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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

#include <wtf/PrintStream.h>
#include <wtf/text/CString.h>

namespace JSC { namespace Profiler {

class Database;

#define JSC_PROFILER_OBJECT_KEYS(macro) \
    macro(bytecode) \
    macro(bytecodeIndex) \
    macro(bytecodes) \
    macro(bytecodesID) \
    macro(counters) \
    macro(opcode) \
    macro(description) \
    macro(descriptions) \
    macro(hash) \
    macro(inferredName) \
    macro(sourceCode) \
    macro(instructionCount) \
    macro(compilationKind) \
    macro(compilationUID) \
    macro(compilations) \
    macro(profiledBytecodes) \
    macro(origin) \
    macro(osrExitSites) \
    macro(osrExits) \
    macro(executionCount) \
    macro(exitKind) \
    macro(numInlinedCalls) \
    macro(numInlinedGetByIds) \
    macro(numInlinedPutByIds) \
    macro(additionalJettisonReason) \
    macro(jettisonReason) \
    macro(uid) \
    macro(events) \
    macro(summary) \
    macro(isWatchpoint) \
    macro(detail) \
    macro(time) \
    macro(id) \
    macro(header) \
    macro(count) \


class Dumper {
public:
    class Keys {
    public:
#define JSC_DEFINE_PROFILER_OBJECT_KEY(key) String m_##key { #key ""_s };
        JSC_PROFILER_OBJECT_KEYS(JSC_DEFINE_PROFILER_OBJECT_KEY)
#undef JSC_DEFINE_PROFILER_OBJECT_KEY
    };

    Dumper(const Database& database)
        : m_database(database)
    {
    }

    const Database& database() const { return m_database; }
    const Keys& keys() const { return m_keys; }

private:
    const Database& m_database;
    Keys m_keys;
};

} } // namespace JSC::Profiler
