/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "ProfilerCompilation.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProfilerDatabase.h"
#include "ProfilerDumper.h"
#include <wtf/StringPrintStream.h>

namespace JSC { namespace Profiler {

Compilation::Compilation(Bytecodes* bytecodes, CompilationKind kind)
    : m_kind(kind)
    , m_bytecodes(bytecodes)
    , m_numInlinedGetByIds(0)
    , m_numInlinedPutByIds(0)
    , m_numInlinedCalls(0)
    , m_jettisonReason(NotJettisoned)
    , m_uid(UID::generate())
{
}

Compilation::~Compilation() = default;

void Compilation::addProfiledBytecodes(Database& database, CodeBlock* profiledBlock)
{
    Bytecodes* bytecodes = database.ensureBytecodesFor(profiledBlock);
    
    // First make sure that we haven't already added profiled bytecodes for this code
    // block. We do this using an O(N) search because I suspect that this list will
    // tend to be fairly small, and the additional space costs of having a UncheckedKeyHashMap/Set
    // would be greater than the time cost of occasionally doing this search.
    
    for (unsigned i = m_profiledBytecodes.size(); i--;) {
        if (m_profiledBytecodes[i].bytecodes() == bytecodes)
            return;
    }
    
    m_profiledBytecodes.append(ProfiledBytecodes(bytecodes, profiledBlock));
}

void Compilation::addDescription(const CompiledBytecode& compiledBytecode)
{
    m_descriptions.append(compiledBytecode);
}

void Compilation::addDescription(const OriginStack& stack, const CString& description)
{
    addDescription(CompiledBytecode(stack, description));
}

ExecutionCounter* Compilation::executionCounterFor(const OriginStack& origin)
{
    std::unique_ptr<ExecutionCounter>& counter = m_counters.add(origin, nullptr).iterator->value;
    if (!counter)
        counter = makeUnique<ExecutionCounter>();

    return counter.get();
}

void Compilation::addOSRExitSite(const Vector<CodePtr<JSInternalPtrTag>>& codeAddresses)
{
    m_osrExitSites.append(OSRExitSite(codeAddresses));
}

OSRExit* Compilation::addOSRExit(unsigned id, const OriginStack& originStack, ExitKind exitKind, bool isWatchpoint)
{
    m_osrExits.append(OSRExit(id, originStack, exitKind, isWatchpoint));
    return &m_osrExits.last();
}

void Compilation::setJettisonReason(JettisonReason jettisonReason, const FireDetail* detail)
{
    if (m_jettisonReason != NotJettisoned)
        return; // We only care about the original jettison reason.
    
    m_jettisonReason = jettisonReason;
    if (detail)
        m_additionalJettisonReason = toCString(*detail);
    else
        m_additionalJettisonReason = CString();
}

void Compilation::dump(PrintStream& out) const
{
    out.print("Comp", m_uid);
}

Ref<JSON::Value> Compilation::toJSON(Dumper& dumper) const
{
    auto result = JSON::Object::create();
    result->setDouble(dumper.keys().m_bytecodesID, m_bytecodes->id());
    result->setString(dumper.keys().m_compilationKind, String::fromUTF8(toCString(m_kind).span()));

    auto profiledBytecodes = JSON::Array::create();
    for (const auto& bytecode : m_profiledBytecodes)
        profiledBytecodes->pushValue(bytecode.toJSON(dumper));
    result->setValue(dumper.keys().m_profiledBytecodes, WTFMove(profiledBytecodes));

    auto descriptions = JSON::Array::create();
    for (const auto& description : m_descriptions)
        descriptions->pushValue(description.toJSON(dumper));
    result->setValue(dumper.keys().m_descriptions, WTFMove(descriptions));

    auto counters = JSON::Array::create();
    for (const auto& [key, value] : m_counters) {
        auto counterEntry = JSON::Object::create();
        counterEntry->setValue(dumper.keys().m_origin, key.toJSON(dumper));
        counterEntry->setDouble(dumper.keys().m_executionCount, value->count());
        counters->pushValue(WTFMove(counterEntry));
    }
    result->setValue(dumper.keys().m_counters, WTFMove(counters));

    auto exitSites = JSON::Array::create();
    for (const auto& osrExitSite : m_osrExitSites)
        exitSites->pushValue(osrExitSite.toJSON(dumper));
    result->setValue(dumper.keys().m_osrExitSites, WTFMove(exitSites));

    auto exits = JSON::Array::create();
    for (unsigned i = 0; i < m_osrExits.size(); ++i)
        exits->pushValue(m_osrExits[i].toJSON(dumper));
    result->setValue(dumper.keys().m_osrExits, WTFMove(exits));

    result->setDouble(dumper.keys().m_numInlinedGetByIds, m_numInlinedGetByIds);
    result->setDouble(dumper.keys().m_numInlinedPutByIds, m_numInlinedPutByIds);
    result->setDouble(dumper.keys().m_numInlinedCalls, m_numInlinedCalls);
    result->setString(dumper.keys().m_jettisonReason, String::fromUTF8(toCString(m_jettisonReason).span()));
    if (!m_additionalJettisonReason.isNull())
        result->setString(dumper.keys().m_additionalJettisonReason, String::fromUTF8(m_additionalJettisonReason.span()));

    result->setString(dumper.keys().m_uid, makeString(m_uid));

    return result;
}

} } // namespace JSC::Profiler

