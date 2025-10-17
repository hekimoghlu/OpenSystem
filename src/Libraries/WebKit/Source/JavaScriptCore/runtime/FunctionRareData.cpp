/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include "FunctionRareData.h"

#include "JSCInlines.h"
#include "ObjectAllocationProfileInlines.h"

namespace JSC {

const ClassInfo FunctionRareData::s_info = { "FunctionRareData"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(FunctionRareData) };

FunctionRareData* FunctionRareData::create(VM& vm, ExecutableBase* executable)
{
    FunctionRareData* rareData = new (NotNull, allocateCell<FunctionRareData>(vm)) FunctionRareData(vm, executable);
    rareData->finishCreation(vm);
    return rareData;
}

void FunctionRareData::destroy(JSCell* cell)
{
    FunctionRareData* rareData = static_cast<FunctionRareData*>(cell);
    rareData->FunctionRareData::~FunctionRareData();
}

Structure* FunctionRareData::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(CellType, StructureFlags), info());
}

template<typename Visitor>
void FunctionRareData::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    FunctionRareData* rareData = jsCast<FunctionRareData*>(cell);
    ASSERT_GC_OBJECT_INHERITS(cell, info());
    Base::visitChildren(cell, visitor);

    rareData->m_objectAllocationProfile.visitAggregate(visitor);
    rareData->m_internalFunctionAllocationProfile.visitAggregate(visitor);
    visitor.append(rareData->m_boundFunctionStructureID);
    visitor.append(rareData->m_executable);
}

DEFINE_VISIT_CHILDREN(FunctionRareData);

FunctionRareData::FunctionRareData(VM& vm, ExecutableBase* executable)
    : Base(vm, vm.functionRareDataStructure.get())
    , m_objectAllocationProfile()
    // We initialize blind so that changes to the prototype after function creation but before
    // the first allocation don't disable optimizations. This isn't super important, since the
    // function is unlikely to allocate a rare data until the first allocation anyway.
    , m_allocationProfileWatchpointSet(ClearWatchpoint)
    , m_executable(executable, WriteBarrierEarlyInit)
    , m_hasReifiedLength(false)
    , m_hasReifiedName(false)
    , m_hasModifiedLengthForBoundOrNonHostFunction(false)
    , m_hasModifiedNameForBoundOrNonHostFunction(false)
{
}

FunctionRareData::~FunctionRareData() = default;

void FunctionRareData::initializeObjectAllocationProfile(VM& vm, JSGlobalObject* globalObject, JSObject* prototype, size_t inlineCapacity, JSFunction* constructor)
{
    initializeAllocationProfileWatchpointSet();
    m_objectAllocationProfile.initializeProfile(vm, globalObject, this, prototype, inlineCapacity, constructor, this);
}

void FunctionRareData::clear(const char* reason)
{
    m_objectAllocationProfile.clear();
    m_internalFunctionAllocationProfile.clear();
    m_allocationProfileWatchpointSet.fireAll(vm(), reason);
}

void FunctionRareData::AllocationProfileClearingWatchpoint::fireInternal(VM&, const FireDetail&)
{
    m_rareData->clear("AllocationProfileClearingWatchpoint fired.");
}

}
