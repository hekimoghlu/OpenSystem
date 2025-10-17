/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include "Butterfly.h"
#include "IndexingHeader.h"
#include "JSCJSValueInlines.h"
#include "JSCell.h"
#include "ResourceExhaustion.h"
#include "Structure.h"
#include "VirtualRegister.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class ClonedArguments;
class DirectArguments;
class ScopedArguments;

class JSImmutableButterfly : public JSCell {
    using Base = JSCell;

public:
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue, IndexingType);

    ALWAYS_INLINE static JSImmutableButterfly* tryCreate(VM& vm, Structure* structure, unsigned length)
    {
        if (UNLIKELY(length > IndexingHeader::maximumLength))
            return nullptr;

        // Because of the above maximumLength requirement, allocationSize can never overflow.
        void* buffer = tryAllocateCell<JSImmutableButterfly>(vm, allocationSize(length));
        if (UNLIKELY(!buffer))
            return nullptr;
        JSImmutableButterfly* result = new (NotNull, buffer) JSImmutableButterfly(vm, structure, length);
        result->finishCreation(vm);
        return result;
    }

    static JSImmutableButterfly* tryCreate(VM& vm, IndexingType indexingType, unsigned length)
    {
        return tryCreate(vm, vm.immutableButterflyStructures[arrayIndexFromIndexingType(indexingType) - NumberOfIndexingShapes].get(), length);
    }

    static JSImmutableButterfly* create(VM& vm, IndexingType indexingType, unsigned length)
    {
        auto* array = tryCreate(vm, indexingType, length);
        RELEASE_ASSERT_RESOURCE_AVAILABLE(array, MemoryExhaustion, "Crash intentionally because memory is exhausted.");
        return array;
    }

    ALWAYS_INLINE static JSImmutableButterfly* createFromArray(JSGlobalObject* globalObject, VM& vm, JSArray* array)
    {
        auto throwScope = DECLARE_THROW_SCOPE(vm);

        IndexingType indexingType = array->indexingType() & IndexingShapeMask;
        unsigned length = array->length();

        // FIXME: JSImmutableButterfly::createFromArray should support re-using non contiguous indexing types as well.
        if (isCopyOnWrite(indexingType)) {
            if (hasContiguous(indexingType))
                return JSImmutableButterfly::fromButterfly(array->butterfly());
        }

        JSImmutableButterfly* result = JSImmutableButterfly::tryCreate(vm, vm.immutableButterflyStructures[arrayIndexFromIndexingType(CopyOnWriteArrayWithContiguous) - NumberOfIndexingShapes].get(), length);
        if (UNLIKELY(!result)) {
            throwOutOfMemoryError(globalObject, throwScope);
            return nullptr;
        }

        if (!length)
            return result;

        if (indexingType == ContiguousShape || indexingType == Int32Shape) {
            for (unsigned i = 0; i < length; i++) {
                JSValue value = array->butterfly()->contiguous().at(array, i).get();
                value = !!value ? value : jsUndefined();
                result->setIndex(vm, i, value);
            }
            return result;
        }

        if (indexingType == DoubleShape) {
            ASSERT(Options::allowDoubleShape());
            for (unsigned i = 0; i < length; i++) {
                double d = array->butterfly()->contiguousDouble().at(array, i);
                JSValue value = std::isnan(d) ? jsUndefined() : JSValue(JSValue::EncodeAsDouble, d);
                result->setIndex(vm, i, value);
            }
            return result;
        }

        for (unsigned i = 0; i < length; i++) {
            JSValue value = array->getDirectIndex(globalObject, i);
            if (!value) {
                // When we see a hole, we assume that it's safe to assume the get would have returned undefined.
                // We may still call into this function when !globalObject->isArrayIteratorProtocolFastAndNonObservable(),
                // however, if we do that, we ensure we're calling in with an array with all self properties between
                // [0, length).
                //
                // We may also call into this during OSR exit to materialize a phantom fixed array.
                // We may be creating a fixed array during OSR exit even after the iterator protocol changed.
                // But, when the phantom would have logically been created, the protocol hadn't been
                // changed. Therefore, it is sound to assume empty indices are jsUndefined().
                value = jsUndefined();
            }
            RETURN_IF_EXCEPTION(throwScope, nullptr);
            result->setIndex(vm, i, value);
        }
        return result;
    }

    static JSImmutableButterfly* createFromClonedArguments(JSGlobalObject*, ClonedArguments*);
    static JSImmutableButterfly* createFromDirectArguments(JSGlobalObject*, DirectArguments*);
    static JSImmutableButterfly* createFromScopedArguments(JSGlobalObject*, ScopedArguments*);
    static JSImmutableButterfly* createFromString(JSGlobalObject*, JSString*);
    static JSImmutableButterfly* tryCreateFromArgList(VM&, ArgList);

    unsigned publicLength() const { return m_header.publicLength(); }
    unsigned vectorLength() const { return m_header.vectorLength(); }
    unsigned length() const { return m_header.publicLength(); }

    Butterfly* toButterfly() const { return std::bit_cast<Butterfly*>(std::bit_cast<char*>(this) + offsetOfData()); }
    static JSImmutableButterfly* fromButterfly(Butterfly* butterfly) { return std::bit_cast<JSImmutableButterfly*>(std::bit_cast<char*>(butterfly) - offsetOfData()); }

    JSValue get(unsigned index) const
    {
        if (!hasDouble(indexingMode()))
            return toButterfly()->contiguous().at(this, index).get();
        double value = toButterfly()->contiguousDouble().at(this, index);
        // Holes are not supported yet.
        ASSERT(!std::isnan(value));
        return jsDoubleNumber(value);
    }

    DECLARE_VISIT_CHILDREN;

    void copyToArguments(JSGlobalObject*, JSValue* firstElementDest, unsigned offset, unsigned length);

    template<typename, SubspaceAccess>
    static CompleteSubspace* subspaceFor(VM& vm)
    {
        // We allocate out of the JSValue gigacage as other code expects all butterflies to live there.
        return &vm.immutableButterflyAuxiliarySpace();
    }

    // Only call this if you just allocated this butterfly.
    void setIndex(VM& vm, unsigned index, JSValue value)
    {
        if (hasDouble(indexingType()))
            toButterfly()->contiguousDouble().atUnsafe(index) = value.asNumber();
        else
            toButterfly()->contiguous().atUnsafe(index).set(vm, this, value);
    }

    static constexpr size_t offsetOfData()
    {
        return WTF::roundUpToMultipleOf<sizeof(WriteBarrier<Unknown>)>(sizeof(JSImmutableButterfly));
    }

    static constexpr ptrdiff_t offsetOfPublicLength()
    {
        return OBJECT_OFFSETOF(JSImmutableButterfly, m_header) + IndexingHeader::offsetOfPublicLength();
    }

    static constexpr ptrdiff_t offsetOfVectorLength()
    {
        return OBJECT_OFFSETOF(JSImmutableButterfly, m_header) + IndexingHeader::offsetOfVectorLength();
    }

    static Checked<size_t> allocationSize(Checked<size_t> numItems)
    {
        return offsetOfData() + numItems * sizeof(WriteBarrier<Unknown>);
    }

private:
    JSImmutableButterfly(VM& vm, Structure* structure, unsigned length)
        : Base(vm, structure)
    {
        m_header.setVectorLength(length);
        m_header.setPublicLength(length);
        if (hasContiguous(indexingType())) {
            for (unsigned index = 0; index < length; ++index)
                toButterfly()->contiguous().at(this, index).setStartingValue(JSValue());
        }
    }

    IndexingHeader m_header;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
