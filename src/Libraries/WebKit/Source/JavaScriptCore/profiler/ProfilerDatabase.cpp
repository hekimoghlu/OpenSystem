/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include "ProfilerDatabase.h"

#include "CodeBlock.h"
#include "JSCInlines.h"
#include "JSONObject.h"
#include "ObjectConstructor.h"
#include "ProfilerDumper.h"
#include <wtf/FilePrintStream.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace Profiler {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Database);

static Lock registrationLock;
static Database* firstDatabase;

Database::Database(VM& vm)
    : m_databaseID(DatabaseID::generate())
    , m_vm(vm)
    , m_shouldSaveAtExit(false)
    , m_nextRegisteredDatabase(nullptr)
{
}

Database::~Database()
{
    if (m_shouldSaveAtExit) {
        removeDatabaseFromAtExit();
        performAtExitSave();
    }
}

Bytecodes* Database::ensureBytecodesFor(CodeBlock* codeBlock)
{
    Locker locker { m_lock };
    return ensureBytecodesFor(locker, codeBlock);
}

Bytecodes* Database::ensureBytecodesFor(const AbstractLocker&, CodeBlock* codeBlock)
{
    codeBlock = codeBlock->baselineAlternative();
    
    UncheckedKeyHashMap<CodeBlock*, Bytecodes*>::iterator iter = m_bytecodesMap.find(codeBlock);
    if (iter != m_bytecodesMap.end())
        return iter->value;
    
    m_bytecodes.append(Bytecodes(m_bytecodes.size(), codeBlock));
    Bytecodes* result = &m_bytecodes.last();
    
    m_bytecodesMap.add(codeBlock, result);
    
    return result;
}

void Database::notifyDestruction(CodeBlock* codeBlock)
{
    Locker locker { m_lock };
    
    m_bytecodesMap.remove(codeBlock);
    m_compilationMap.remove(codeBlock);
}

void Database::addCompilation(CodeBlock* codeBlock, Ref<Compilation>&& compilation)
{
    Locker locker { m_lock };

    m_compilations.append(compilation.copyRef());
    m_compilationMap.set(codeBlock, WTFMove(compilation));
}

Ref<JSON::Value> Database::toJSON() const
{
    Dumper dumper(*this);
    auto result = JSON::Object::create();

    auto bytecodes = JSON::Array::create();
    for (unsigned i = 0; i < m_bytecodes.size(); ++i)
        bytecodes->pushValue(m_bytecodes[i].toJSON(dumper));
    result->setValue(dumper.keys().m_bytecodes, WTFMove(bytecodes));

    auto compilations = JSON::Array::create();
    for (unsigned i = 0; i < m_compilations.size(); ++i)
        compilations->pushValue(m_compilations[i]->toJSON(dumper));
    result->setValue(dumper.keys().m_compilations, WTFMove(compilations));

    auto events = JSON::Array::create();
    for (unsigned i = 0; i < m_events.size(); ++i)
        events->pushValue(m_events[i].toJSON(dumper));
    result->setValue(dumper.keys().m_events, WTFMove(events));

    return result;
}

bool Database::save(const char* filename) const
{
    auto out = FilePrintStream::open(filename, "w");
    if (!out)
        return false;
    out->print(toJSON().get());
    return true;
}

void Database::registerToSaveAtExit(const char* filename)
{
    m_atExitSaveFilename = filename;
    
    if (m_shouldSaveAtExit)
        return;
    
    addDatabaseToAtExit();
    m_shouldSaveAtExit = true;
}

void Database::logEvent(CodeBlock* codeBlock, const char* summary, const CString& detail)
{
    Locker locker { m_lock };
    
    Bytecodes* bytecodes = ensureBytecodesFor(locker, codeBlock);
    Compilation* compilation = m_compilationMap.get(codeBlock);
    m_events.append(Event(WallTime::now(), bytecodes, compilation, summary, detail));
}

void Database::addDatabaseToAtExit()
{
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        atexit(atExitCallback);
    });

    {
        Locker locker { registrationLock };
        m_nextRegisteredDatabase = firstDatabase;
        firstDatabase = this;
    }
}

void Database::removeDatabaseFromAtExit()
{
    Locker locker { registrationLock };
    for (Database** current = &firstDatabase; *current; current = &(*current)->m_nextRegisteredDatabase) {
        if (*current != this)
            continue;
        *current = m_nextRegisteredDatabase;
        m_nextRegisteredDatabase = nullptr;
        m_shouldSaveAtExit = false;
        break;
    }
}

void Database::performAtExitSave() const
{
    JSLockHolder lock(m_vm);
    save(m_atExitSaveFilename.data());
}

Database* Database::removeFirstAtExitDatabase()
{
    Locker locker { registrationLock };
    Database* result = firstDatabase;
    if (result) {
        firstDatabase = result->m_nextRegisteredDatabase;
        result->m_nextRegisteredDatabase = nullptr;
        result->m_shouldSaveAtExit = false;
    }
    return result;
}

void Database::atExitCallback()
{
    while (Database* database = removeFirstAtExitDatabase())
        database->performAtExitSave();
}

} } // namespace JSC::Profiler

