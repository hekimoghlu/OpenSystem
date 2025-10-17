/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "ScopedArgumentsTable.h"

#include "JSCJSValueInlines.h"
#include "JSObjectInlines.h"
#include "StructureInlines.h"

namespace JSC {

const ClassInfo ScopedArgumentsTable::s_info = { "ScopedArgumentsTable"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(ScopedArgumentsTable) };

ScopedArgumentsTable::ScopedArgumentsTable(VM& vm)
    : Base(vm, vm.scopedArgumentsTableStructure.get())
    , m_length(0)
    , m_locked(false)
{
}

ScopedArgumentsTable::~ScopedArgumentsTable() = default;

void ScopedArgumentsTable::destroy(JSCell* cell)
{
    static_cast<ScopedArgumentsTable*>(cell)->ScopedArgumentsTable::~ScopedArgumentsTable();
}

ScopedArgumentsTable* ScopedArgumentsTable::create(VM& vm)
{
    ScopedArgumentsTable* result =
        new (NotNull, allocateCell<ScopedArgumentsTable>(vm)) ScopedArgumentsTable(vm);
    result->finishCreation(vm);
    return result;
}

ScopedArgumentsTable* ScopedArgumentsTable::tryCreate(VM& vm, uint32_t length)
{
    void* buffer = tryAllocateCell<ScopedArgumentsTable>(vm);
    if (UNLIKELY(!buffer))
        return nullptr;
    ScopedArgumentsTable* result = new (NotNull, buffer) ScopedArgumentsTable(vm);
    result->finishCreation(vm);

    result->m_length = length;
    result->m_arguments = ArgumentsPtr::tryCreate(length);
    if (UNLIKELY(!result->m_arguments))
        return nullptr;
    result->m_watchpointSets.fill(nullptr, length);
    return result;
}

ScopedArgumentsTable* ScopedArgumentsTable::tryClone(VM& vm)
{
    ScopedArgumentsTable* result = tryCreate(vm, m_length);
    if (UNLIKELY(!result))
        return nullptr;
    for (unsigned i = m_length; i--;)
        result->at(i) = this->at(i);
    result->m_watchpointSets = this->m_watchpointSets;
    return result;
}

ScopedArgumentsTable* ScopedArgumentsTable::trySetLength(VM& vm, uint32_t newLength)
{
    if (LIKELY(!m_locked)) {
        ArgumentsPtr newArguments = ArgumentsPtr::tryCreate(newLength, newLength);
        if (UNLIKELY(!newArguments))
            return nullptr;
        for (unsigned i = std::min(m_length, newLength); i--;)
            newArguments.at(i) = this->at(i);
        m_length = newLength;
        m_arguments = WTFMove(newArguments);
        m_watchpointSets.resize(newLength);
        return this;
    }
    
    ScopedArgumentsTable* result = tryCreate(vm, newLength);
    if (UNLIKELY(!result))
        return nullptr;
    m_watchpointSets.resize(newLength);
    for (unsigned i = std::min(m_length, newLength); i--;) {
        result->at(i) = this->at(i);
        result->m_watchpointSets[i] = this->m_watchpointSets[i];
    }
    return result;
}

static_assert(std::is_trivially_destructible<ScopeOffset>::value);

ScopedArgumentsTable* ScopedArgumentsTable::trySet(VM& vm, uint32_t i, ScopeOffset value)
{
    ScopedArgumentsTable* result;
    if (UNLIKELY(m_locked)) {
        result = tryClone(vm);
        if (UNLIKELY(!result))
            return nullptr;
    } else
        result = this;
    result->at(i) = value;
    return result;
}

void ScopedArgumentsTable::trySetWatchpointSet(uint32_t i, WatchpointSet* watchpoints)
{
    if (!watchpoints)
        return;

    if (i >= m_watchpointSets.size())
        return;

    m_watchpointSets[i] = watchpoints;
}

Structure* ScopedArgumentsTable::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(CellType, StructureFlags), info());
}

} // namespace JSC

