/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#include "DirectArguments.h"

#include "CodeBlock.h"
#include "GenericArgumentsInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(DirectArguments);

const ClassInfo DirectArguments::s_info = { "Arguments"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(DirectArguments) };

DirectArguments::DirectArguments(VM& vm, Structure* structure, unsigned length, unsigned capacity)
    : GenericArguments(vm, structure)
    , m_length(length)
    , m_minCapacity(capacity)
{
    // When we construct the object from C++ code, we expect the capacity to be at least as large as
    // length. JIT-allocated DirectArguments objects play evil tricks, though.
    ASSERT(capacity >= length);
}

DirectArguments* DirectArguments::createUninitialized(
    VM& vm, Structure* structure, unsigned length, unsigned capacity)
{
    DirectArguments* result =
        new (NotNull, allocateCell<DirectArguments>(vm, allocationSize(capacity)))
        DirectArguments(vm, structure, length, capacity);
    result->finishCreation(vm);
    return result;
}

DirectArguments* DirectArguments::create(VM& vm, Structure* structure, unsigned length, unsigned capacity)
{
    DirectArguments* result = createUninitialized(vm, structure, length, capacity);
    
    for (unsigned i = capacity; i--;)
        result->storage()[i].setUndefined();
    
    return result;
}

DirectArguments* DirectArguments::createByCopying(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    VM& vm = globalObject->vm();
    
    unsigned length = callFrame->argumentCount();
    unsigned capacity = std::max(length, static_cast<unsigned>(callFrame->codeBlock()->numParameters() - 1));
    DirectArguments* result = createUninitialized(
        vm, globalObject->directArgumentsStructure(), length, capacity);
    
    for (unsigned i = capacity; i--;)
        result->storage()[i].set(vm, result, callFrame->getArgumentUnsafe(i));
    
    result->setCallee(vm, jsCast<JSFunction*>(callFrame->jsCallee()));
    
    return result;
}

size_t DirectArguments::estimatedSize(JSCell* cell, VM& vm)
{
    DirectArguments* thisObject = jsCast<DirectArguments*>(cell);
    size_t mappedArgumentsSize = thisObject->m_mappedArguments ? thisObject->mappedArgumentsSize() * sizeof(bool) : 0;
    size_t modifiedArgumentsSize = thisObject->m_modifiedArgumentsDescriptor ? thisObject->m_length * sizeof(bool) : 0;
    return Base::estimatedSize(cell, vm) + mappedArgumentsSize + modifiedArgumentsSize;
}

template<typename Visitor>
void DirectArguments::visitChildrenImpl(JSCell* thisCell, Visitor& visitor)
{
    DirectArguments* thisObject = static_cast<DirectArguments*>(thisCell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.appendValues(thisObject->storage(), std::max(thisObject->m_length, thisObject->m_minCapacity));
    visitor.append(thisObject->m_callee);

    if (thisObject->m_mappedArguments)
        visitor.markAuxiliary(thisObject->m_mappedArguments.get());
    GenericArguments<DirectArguments>::visitChildren(thisCell, visitor);
}

DEFINE_VISIT_CHILDREN(DirectArguments);

Structure* DirectArguments::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(DirectArgumentsType, StructureFlags), info());
}

void DirectArguments::overrideThings(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    RELEASE_ASSERT(!m_mappedArguments);
    
    putDirect(vm, vm.propertyNames->length, jsNumber(m_length), static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirect(vm, vm.propertyNames->callee, m_callee.get(), static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirect(vm, vm.propertyNames->iteratorSymbol, globalObject->arrayProtoValuesFunction(), static_cast<unsigned>(PropertyAttribute::DontEnum));
    
    void* backingStore = vm.gigacageAuxiliarySpace(m_mappedArguments.kind).allocate(vm, mappedArgumentsSize(), nullptr, AllocationFailureMode::ReturnNull);
    if (UNLIKELY(!backingStore)) {
        throwOutOfMemoryError(globalObject, scope);
        return;
    }
    bool* overrides = static_cast<bool*>(backingStore);
    m_mappedArguments.set(vm, this, overrides);
    for (unsigned i = internalLength(); i--;)
        overrides[i] = false;
}

void DirectArguments::overrideThingsIfNecessary(JSGlobalObject* globalObject)
{
    if (!m_mappedArguments)
        overrideThings(globalObject);
}

void DirectArguments::unmapArgument(JSGlobalObject* globalObject, unsigned index)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    overrideThingsIfNecessary(globalObject);
    RETURN_IF_EXCEPTION(scope, void());

    m_mappedArguments.at(index) = true;
}

void DirectArguments::copyToArguments(JSGlobalObject* globalObject, JSValue* firstElementDest, unsigned offset, unsigned length)
{
    if (!m_mappedArguments) {
        unsigned limit = std::min(length + offset, m_length);
        unsigned i;
        for (i = offset; i < limit; ++i)
            firstElementDest[i - offset] = storage()[i].get();
        for (; i < length; ++i)
            firstElementDest[i - offset] = get(globalObject, i);
        return;
    }

    GenericArguments::copyToArguments(globalObject, firstElementDest, offset, length);
}

unsigned DirectArguments::mappedArgumentsSize()
{
    // We always allocate something; in the relatively uncommon case of overriding an empty argument we
    // still allocate so that m_mappedArguments is non-null. We use that to indicate that the other properties
    // (length, etc) are overridden.
    return WTF::roundUpToMultipleOf<8>(m_length ? m_length : 1);
}

bool DirectArguments::isIteratorProtocolFastAndNonObservable()
{
    Structure* structure = this->structure();
    JSGlobalObject* globalObject = structure->globalObject();
    if (!globalObject->isArgumentsPrototypeIteratorProtocolFastAndNonObservable())
        return false;

    if (UNLIKELY(m_mappedArguments))
        return false;

    if (structure->didTransition())
        return false;

    return true;
}

JSArray* DirectArguments::fastSlice(JSGlobalObject* globalObject, DirectArguments* arguments, uint64_t startIndex, uint64_t count)
{
    if (count >= MIN_SPARSE_ARRAY_INDEX)
        return nullptr;

    if (UNLIKELY(arguments->m_mappedArguments))
        return nullptr;

    if (startIndex + count > arguments->m_length)
        return nullptr;

    Structure* resultStructure = globalObject->arrayStructureForIndexingTypeDuringAllocation(ArrayWithContiguous);
    if (UNLIKELY(hasAnyArrayStorage(resultStructure->indexingType())))
        return nullptr;

    ObjectInitializationScope scope(globalObject->vm());
    JSArray* resultArray = JSArray::tryCreateUninitializedRestricted(scope, resultStructure, static_cast<uint32_t>(count));
    if (UNLIKELY(!resultArray))
        return nullptr;

    auto& resultButterfly = *resultArray->butterfly();
    gcSafeMemcpy(resultButterfly.contiguous().data(), arguments->storage() + startIndex, sizeof(JSValue) * static_cast<uint32_t>(count));

    ASSERT(resultButterfly.publicLength() == count);
    return resultArray;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
