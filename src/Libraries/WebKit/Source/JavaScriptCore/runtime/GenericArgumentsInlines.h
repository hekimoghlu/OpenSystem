/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "GenericArguments.h"
#include "JSCInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template<typename Type>
template<typename Visitor>
void GenericArguments<Type>::visitChildrenImpl(JSCell* thisCell, Visitor& visitor)
{
    Type* thisObject = static_cast<Type*>(thisCell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisCell, visitor);
    
    if (thisObject->m_modifiedArgumentsDescriptor)
        visitor.markAuxiliary(thisObject->m_modifiedArgumentsDescriptor.getUnsafe());
}

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(template<typename Type>, GenericArguments<Type>);

template<typename Type>
bool GenericArguments<Type>::getOwnPropertySlot(JSObject* object, JSGlobalObject* globalObject, PropertyName ident, PropertySlot& slot)
{
    Type* thisObject = jsCast<Type*>(object);
    VM& vm = globalObject->vm();
    
    if (!thisObject->overrodeThings()) {
        if (ident == vm.propertyNames->length) {
            slot.setValue(thisObject, static_cast<unsigned>(PropertyAttribute::DontEnum), jsNumber(thisObject->internalLength()));
            return true;
        }
        if (ident == vm.propertyNames->callee) {
            slot.setValue(thisObject, static_cast<unsigned>(PropertyAttribute::DontEnum), thisObject->callee());
            return true;
        }
        if (ident == vm.propertyNames->iteratorSymbol) {
            slot.setValue(thisObject, static_cast<unsigned>(PropertyAttribute::DontEnum), thisObject->globalObject()->arrayProtoValuesFunction());
            return true;
        }
    }
    
    if (std::optional<uint32_t> index = parseIndex(ident))
        return GenericArguments<Type>::getOwnPropertySlotByIndex(thisObject, globalObject, *index, slot);

    return Base::getOwnPropertySlot(thisObject, globalObject, ident, slot);
}

template<typename Type>
bool GenericArguments<Type>::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* globalObject, unsigned index, PropertySlot& slot)
{
    Type* thisObject = jsCast<Type*>(object);
    
    if (!thisObject->isModifiedArgumentDescriptor(index) && thisObject->isMappedArgument(index)) {
        slot.setValue(thisObject, static_cast<unsigned>(PropertyAttribute::None), thisObject->getIndexQuickly(index));
        return true;
    }
    
    bool result = Base::getOwnPropertySlotByIndex(object, globalObject, index, slot);
    
    if (thisObject->isMappedArgument(index)) {
        ASSERT(result);
        slot.setValue(thisObject, slot.attributes(), thisObject->getIndexQuickly(index));
        return true;
    }
    
    return result;
}

template<typename Type>
void GenericArguments<Type>::getOwnPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& array, DontEnumPropertiesMode mode)
{
    VM& vm = globalObject->vm();
    Type* thisObject = jsCast<Type*>(object);

    if (array.includeStringProperties()) {
        for (unsigned i = 0; i < thisObject->internalLength(); ++i) {
            if (!thisObject->isMappedArgument(i))
                continue;
            array.add(Identifier::from(vm, i));
        }
        thisObject->getOwnIndexedPropertyNames(globalObject, array, mode);
    }

    if (mode == DontEnumPropertiesMode::Include && !thisObject->overrodeThings()) {
        array.add(vm.propertyNames->length);
        array.add(vm.propertyNames->callee);
        array.add(vm.propertyNames->iteratorSymbol);
    }
    thisObject->getOwnNonIndexPropertyNames(globalObject, array, mode);
}

template<typename Type>
bool GenericArguments<Type>::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName ident, JSValue value, PutPropertySlot& slot)
{
    Type* thisObject = jsCast<Type*>(cell);
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    
    if (!thisObject->overrodeThings()
        && (ident == vm.propertyNames->length
            || ident == vm.propertyNames->callee
            || ident == vm.propertyNames->iteratorSymbol)) {
        thisObject->overrideThings(globalObject);
        RETURN_IF_EXCEPTION(scope, false);
        PutPropertySlot dummy = slot; // This put is not cacheable, so we shadow the slot that was given to us.
        RELEASE_AND_RETURN(scope, Base::put(thisObject, globalObject, ident, value, dummy));
    }

    // https://tc39.github.io/ecma262/#sec-arguments-exotic-objects-set-p-v-receiver
    // Fall back to the OrdinarySet when the receiver is altered from the thisObject.
    if (UNLIKELY(slot.thisValue() != thisObject))
        RELEASE_AND_RETURN(scope, Base::put(thisObject, globalObject, ident, value, slot));
    
    std::optional<uint32_t> index = parseIndex(ident);
    if (index && thisObject->isMappedArgument(index.value())) {
        thisObject->setIndexQuickly(vm, index.value(), value);
        return true;
    }

    RELEASE_AND_RETURN(scope, Base::put(thisObject, globalObject, ident, value, slot));
}

template<typename Type>
bool GenericArguments<Type>::putByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned index, JSValue value, bool shouldThrow)
{
    Type* thisObject = jsCast<Type*>(cell);
    VM& vm = globalObject->vm();

    if (thisObject->isMappedArgument(index)) {
        thisObject->setIndexQuickly(vm, index, value);
        return true;
    }
    
    return Base::putByIndex(cell, globalObject, index, value, shouldThrow);
}

template<typename Type>
bool GenericArguments<Type>::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName ident, DeletePropertySlot& slot)
{
    Type* thisObject = jsCast<Type*>(cell);
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    
    if (!thisObject->overrodeThings()
        && (ident == vm.propertyNames->length
            || ident == vm.propertyNames->callee
            || ident == vm.propertyNames->iteratorSymbol)) {
        thisObject->overrideThings(globalObject);
        RETURN_IF_EXCEPTION(scope, false);
    }

    if (std::optional<uint32_t> index = parseIndex(ident))
        RELEASE_AND_RETURN(scope, GenericArguments<Type>::deletePropertyByIndex(thisObject, globalObject, *index));

    RELEASE_AND_RETURN(scope, Base::deleteProperty(thisObject, globalObject, ident, slot));
}

template<typename Type>
bool GenericArguments<Type>::deletePropertyByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned index)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    Type* thisObject = jsCast<Type*>(cell);

    bool propertyMightBeInJSObjectStorage = thisObject->isModifiedArgumentDescriptor(index) || !thisObject->isMappedArgument(index);
    bool deletedProperty = true;
    if (propertyMightBeInJSObjectStorage) {
        deletedProperty = Base::deletePropertyByIndex(cell, globalObject, index);
        RETURN_IF_EXCEPTION(scope, true);
    }

    if (deletedProperty) {
        // Deleting an indexed property unconditionally unmaps it.
        if (thisObject->isMappedArgument(index)) {
            // We need to check that the property was mapped so we don't write to random memory.
            thisObject->unmapArgument(globalObject, index);
            RETURN_IF_EXCEPTION(scope, true);
        }
        thisObject->setModifiedArgumentDescriptor(globalObject, index);
        RETURN_IF_EXCEPTION(scope, true);
    }

    return deletedProperty;
}

// https://tc39.es/ecma262/#sec-arguments-exotic-objects-defineownproperty-p-desc
template<typename Type>
bool GenericArguments<Type>::defineOwnProperty(JSObject* object, JSGlobalObject* globalObject, PropertyName ident, const PropertyDescriptor& descriptor, bool shouldThrow)
{
    Type* thisObject = jsCast<Type*>(object);
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    
    if (ident == vm.propertyNames->length
        || ident == vm.propertyNames->callee
        || ident == vm.propertyNames->iteratorSymbol) {
        thisObject->overrideThingsIfNecessary(globalObject);
        RETURN_IF_EXCEPTION(scope, false);
    } else if (std::optional<uint32_t> optionalIndex = parseIndex(ident)) {
        uint32_t index = optionalIndex.value();
        bool isMapped = thisObject->isMappedArgument(index);
        PropertyDescriptor newDescriptor = descriptor;

        if (isMapped) {
            if (thisObject->isModifiedArgumentDescriptor(index)) {
                if (!descriptor.value() && descriptor.writablePresent() && !descriptor.writable())
                    newDescriptor.setValue(thisObject->getIndexQuickly(index));
            } else
                thisObject->putDirectIndex(globalObject, index, thisObject->getIndexQuickly(index));

            scope.assertNoException();
        }

        bool status = thisObject->defineOwnIndexedProperty(globalObject, index, newDescriptor, shouldThrow);
        RETURN_IF_EXCEPTION(scope, false);
        if (!status) {
            ASSERT(!isMapped || thisObject->isModifiedArgumentDescriptor(index));
            return false;
        }

        thisObject->setModifiedArgumentDescriptor(globalObject, index);
        RETURN_IF_EXCEPTION(scope, false);

        if (isMapped) {
            if (descriptor.isAccessorDescriptor())
                thisObject->unmapArgument(globalObject, index);
            else {
                if (descriptor.value())
                    thisObject->setIndexQuickly(vm, index, descriptor.value());
                if (descriptor.writablePresent() && !descriptor.writable())
                    thisObject->unmapArgument(globalObject, index);
            }

            RETURN_IF_EXCEPTION(scope, false);
        }

        return true;
    }

    RELEASE_AND_RETURN(scope, Base::defineOwnProperty(object, globalObject, ident, descriptor, shouldThrow));
}

template<typename Type>
void GenericArguments<Type>::initModifiedArgumentsDescriptor(JSGlobalObject* globalObject, unsigned argsLength)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    RELEASE_ASSERT(!m_modifiedArgumentsDescriptor);

    if (argsLength) {
        void* backingStore = vm.gigacageAuxiliarySpace(m_modifiedArgumentsDescriptor.kind).allocate(vm, WTF::roundUpToMultipleOf<8>(argsLength), nullptr, AllocationFailureMode::ReturnNull);
        if (UNLIKELY(!backingStore)) {
            throwOutOfMemoryError(globalObject, scope);
            return;
        }
        bool* modifiedArguments = static_cast<bool*>(backingStore);
        m_modifiedArgumentsDescriptor.set(vm, this, modifiedArguments);
        for (unsigned i = argsLength; i--;)
            modifiedArguments[i] = false;
    }
}

template<typename Type>
void GenericArguments<Type>::initModifiedArgumentsDescriptorIfNecessary(JSGlobalObject* globalObject, unsigned argsLength)
{
    if (!m_modifiedArgumentsDescriptor)
        initModifiedArgumentsDescriptor(globalObject, argsLength);
}

template<typename Type>
void GenericArguments<Type>::setModifiedArgumentDescriptor(JSGlobalObject* globalObject, unsigned index, unsigned length)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    initModifiedArgumentsDescriptorIfNecessary(globalObject, length);
    RETURN_IF_EXCEPTION(scope, void());
    if (index < length)
        m_modifiedArgumentsDescriptor.at(index) = true;
}

template<typename Type>
bool GenericArguments<Type>::isModifiedArgumentDescriptor(unsigned index, unsigned length)
{
    if (!m_modifiedArgumentsDescriptor)
        return false;
    if (index < length)
        return m_modifiedArgumentsDescriptor.at(index);
    return false;
}

template<typename Type>
void GenericArguments<Type>::copyToArguments(JSGlobalObject* globalObject, JSValue* firstElementDest, unsigned offset, unsigned length)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    Type* thisObject = static_cast<Type*>(this);
    for (unsigned i = 0; i < length; ++i) {
        if (thisObject->isMappedArgument(i + offset))
            firstElementDest[i] = thisObject->getIndexQuickly(i + offset);
        else {
            firstElementDest[i] = get(globalObject, i + offset);
            RETURN_IF_EXCEPTION(scope, void());
        }
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
