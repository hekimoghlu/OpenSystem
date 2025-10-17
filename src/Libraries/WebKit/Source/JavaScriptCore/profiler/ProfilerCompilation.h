/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "ExitKind.h"
#include "JSCJSValue.h"
#include "ProfilerCompilationKind.h"
#include "ProfilerCompiledBytecode.h"
#include "ProfilerExecutionCounter.h"
#include "ProfilerJettisonReason.h"
#include "ProfilerOSRExit.h"
#include "ProfilerOSRExitSite.h"
#include "ProfilerOriginStack.h"
#include "ProfilerProfiledBytecodes.h"
#include "ProfilerUID.h"
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/SegmentedVector.h>

namespace JSC {

class FireDetail;

namespace Profiler {

class Bytecodes;
class Database;
class Dumper;

// Represents the act of executing some bytecodes in some engine, and does
// all of the counting for those executions.

class Compilation : public RefCounted<Compilation> {
public:
    Compilation(Bytecodes*, CompilationKind);
    ~Compilation();
    
    void addProfiledBytecodes(Database&, CodeBlock*);
    unsigned profiledBytecodesSize() const { return m_profiledBytecodes.size(); }
    const ProfiledBytecodes& profiledBytecodesAt(unsigned i) const { return m_profiledBytecodes[i]; }
    
    void noticeInlinedGetById() { m_numInlinedGetByIds++; }
    void noticeInlinedPutById() { m_numInlinedPutByIds++; }
    void noticeInlinedCall() { m_numInlinedCalls++; }
    
    Bytecodes* bytecodes() const { return m_bytecodes; }
    CompilationKind kind() const { return m_kind; }
    
    void addDescription(const CompiledBytecode&);
    void addDescription(const OriginStack&, const CString& description);
    ExecutionCounter* executionCounterFor(const OriginStack&);
    void addOSRExitSite(const Vector<CodePtr<JSInternalPtrTag>>& codeAddresses);
    OSRExit* addOSRExit(unsigned id, const OriginStack&, ExitKind, bool isWatchpoint);
    
    void setJettisonReason(JettisonReason, const FireDetail*);
    
    UID uid() const { return m_uid; }
    
    void dump(PrintStream&) const;
    Ref<JSON::Value> toJSON(Dumper&) const;
    
private:
    CompilationKind m_kind;
    Bytecodes* m_bytecodes;
    Vector<ProfiledBytecodes> m_profiledBytecodes;
    Vector<CompiledBytecode> m_descriptions;
    UncheckedKeyHashMap<OriginStack, std::unique_ptr<ExecutionCounter>> m_counters;
    Vector<OSRExitSite> m_osrExitSites;
    SegmentedVector<OSRExit> m_osrExits;
    unsigned m_numInlinedGetByIds;
    unsigned m_numInlinedPutByIds;
    unsigned m_numInlinedCalls;
    JettisonReason m_jettisonReason;
    CString m_additionalJettisonReason;
    UID m_uid;
};

} } // namespace JSC::Profiler
