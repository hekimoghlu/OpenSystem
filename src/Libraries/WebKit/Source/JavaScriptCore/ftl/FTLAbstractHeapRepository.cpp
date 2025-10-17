/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#include "FTLAbstractHeapRepository.h"

#if ENABLE(FTL_JIT)

#include "B3CCallValue.h"
#include "B3FenceValue.h"
#include "B3MemoryValue.h"
#include "B3PatchpointValue.h"
#include "B3ValueInlines.h"
#include "ClonedArguments.h"
#include "DateInstance.h"
#include "DirectArguments.h"
#include "FTLState.h"
#include "GetterSetter.h"
#include "JSPropertyNameEnumerator.h"
#include "JSWebAssemblyInstance.h"
#include "JSWrapperObject.h"
#include "RegExpObject.h"
#include "ScopedArguments.h"
#include "ShadowChicken.h"
#include "StructureChain.h"
#include "StructureRareDataInlines.h"
#include "WebAssemblyModuleRecord.h"

namespace JSC { namespace FTL {

AbstractHeapRepository::AbstractHeapRepository()
    : root(nullptr, "jscRoot")

#define ABSTRACT_HEAP_INITIALIZATION(name) , name(&root, #name)
    FOR_EACH_ABSTRACT_HEAP(ABSTRACT_HEAP_INITIALIZATION)
#undef ABSTRACT_HEAP_INITIALIZATION

#define ABSTRACT_FIELD_INITIALIZATION(name, offset) , name(&root, #name, offset)
    FOR_EACH_ABSTRACT_FIELD(ABSTRACT_FIELD_INITIALIZATION)
#undef ABSTRACT_FIELD_INITIALIZATION
    
    , JSCell_freeListNext(JSCell_header)
    , ArrayStorage_publicLength(Butterfly_publicLength)
    , ArrayStorage_vectorLength(Butterfly_vectorLength)
    
#define INDEXED_ABSTRACT_HEAP_INITIALIZATION(name, offset, size) , name(&root, #name, offset, size)
    FOR_EACH_INDEXED_ABSTRACT_HEAP(INDEXED_ABSTRACT_HEAP_INITIALIZATION)
#undef INDEXED_ABSTRACT_HEAP_INITIALIZATION
    
#define NUMBERED_ABSTRACT_HEAP_INITIALIZATION(name) , name(&root, #name)
    FOR_EACH_NUMBERED_ABSTRACT_HEAP(NUMBERED_ABSTRACT_HEAP_INITIALIZATION)
#undef NUMBERED_ABSTRACT_HEAP_INITIALIZATION

    , JSString_value(JSRopeString_fiber0)
    , JSWrapperObject_internalValue(const_cast<AbstractHeap&>(JSInternalFieldObjectImpl_internalFields[static_cast<unsigned>(JSWrapperObject::Field::WrappedValue)]))

    , absolute(&root, "absolute")
{
    JSCell_header.changeParent(&JSCellHeaderAndNamedProperties);
    properties.atAnyNumber().changeParent(&JSCellHeaderAndNamedProperties);

    // Make sure that our explicit assumptions about the TypeInfoBlob match reality.
    RELEASE_ASSERT(!(JSCell_indexingTypeAndMisc.offset() & (sizeof(int32_t) - 1)));
    RELEASE_ASSERT(JSCell_indexingTypeAndMisc.offset() + 1 == JSCell_typeInfoType.offset());
    RELEASE_ASSERT(JSCell_indexingTypeAndMisc.offset() + 2 == JSCell_typeInfoFlags.offset());
    RELEASE_ASSERT(JSCell_indexingTypeAndMisc.offset() + 3 == JSCell_cellState.offset());

    JSCell_structureID.changeParent(&JSCell_header);
    JSCell_usefulBytes.changeParent(&JSCell_header);
    JSCell_indexingTypeAndMisc.changeParent(&JSCell_usefulBytes);
    JSCell_typeInfoType.changeParent(&JSCell_usefulBytes);
    JSCell_typeInfoFlags.changeParent(&JSCell_usefulBytes);
    JSCell_cellState.changeParent(&JSCell_usefulBytes);
    JSRopeString_flags.changeParent(&JSRopeString_fiber0);
    JSRopeString_length.changeParent(&JSRopeString_fiber1);

    RELEASE_ASSERT(!JSCell_freeListNext.offset());
}

AbstractHeapRepository::~AbstractHeapRepository() = default;

void AbstractHeapRepository::decorateMemory(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForMemory.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decorateCCallRead(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForCCallRead.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decorateCCallWrite(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForCCallWrite.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decoratePatchpointRead(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForPatchpointRead.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decoratePatchpointWrite(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForPatchpointWrite.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decorateFenceRead(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForFenceRead.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decorateFenceWrite(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForFenceWrite.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::decorateFencedAccess(const AbstractHeap* heap, B3::Value* value)
{
    m_heapForFencedAccess.append(HeapForValue(heap, value));
}

void AbstractHeapRepository::computeRangesAndDecorateInstructions()
{
    using namespace B3;
    root.compute();

    if (verboseCompilationEnabled()) {
        dataLog("Abstract Heap Repository:\n");
        root.deepDump(WTF::dataFile());
    }
    
    auto rangeFor = [&] (const AbstractHeap* heap) -> HeapRange {
        if (heap)
            return heap->range();
        return HeapRange();
    };

    for (HeapForValue entry : m_heapForMemory)
        entry.value->as<MemoryValue>()->setRange(rangeFor(entry.heap));
    for (HeapForValue entry : m_heapForCCallRead)
        entry.value->as<CCallValue>()->effects.reads = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForCCallWrite)
        entry.value->as<CCallValue>()->effects.writes = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForPatchpointRead)
        entry.value->as<PatchpointValue>()->effects.reads = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForPatchpointWrite)
        entry.value->as<PatchpointValue>()->effects.writes = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForFenceRead)
        entry.value->as<FenceValue>()->read = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForFenceWrite)
        entry.value->as<FenceValue>()->write = rangeFor(entry.heap);
    for (HeapForValue entry : m_heapForFencedAccess)
        entry.value->as<MemoryValue>()->setFenceRange(rangeFor(entry.heap));
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

