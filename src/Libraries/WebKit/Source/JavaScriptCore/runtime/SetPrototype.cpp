/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "SetPrototype.h"

#include "BuiltinNames.h"
#include "GetterSetter.h"
#include "JSCInlines.h"
#include "JSSet.h"
#include "JSSetIterator.h"

namespace JSC {

const ClassInfo SetPrototype::s_info = { "Set"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(SetPrototype) };

static JSC_DECLARE_HOST_FUNCTION(setProtoFuncAdd);
static JSC_DECLARE_HOST_FUNCTION(setProtoFuncClear);
static JSC_DECLARE_HOST_FUNCTION(setProtoFuncDelete);
static JSC_DECLARE_HOST_FUNCTION(setProtoFuncHas);
static JSC_DECLARE_HOST_FUNCTION(setProtoFuncValues);
static JSC_DECLARE_HOST_FUNCTION(setProtoFuncEntries);

static JSC_DECLARE_HOST_FUNCTION(setProtoFuncSize);

void SetPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSFunction* addFunc = JSFunction::create(vm, globalObject, 1, vm.propertyNames->add.string(), setProtoFuncAdd, ImplementationVisibility::Public, JSSetAddIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->add, addFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().addPrivateName(), addFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* clearFunc = JSFunction::create(vm, globalObject, 0, vm.propertyNames->clear.string(), setProtoFuncClear, ImplementationVisibility::Public);
    putDirectWithoutTransition(vm, vm.propertyNames->clear, clearFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().clearPrivateName(), clearFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* deleteFunc = JSFunction::create(vm, globalObject, 1, vm.propertyNames->deleteKeyword.string(), setProtoFuncDelete, ImplementationVisibility::Public, JSSetDeleteIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->deleteKeyword, deleteFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().deletePrivateName(), deleteFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* entriesFunc = JSFunction::create(vm, globalObject, 0, vm.propertyNames->builtinNames().entriesPublicName().string(), setProtoFuncEntries, ImplementationVisibility::Public, JSSetEntriesIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().entriesPublicName(), entriesFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().entriesPrivateName(), entriesFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* forEachFunc = JSFunction::create(vm, globalObject, setPrototypeForEachCodeGenerator(vm), globalObject);
    putDirectWithoutTransition(vm, vm.propertyNames->forEach, forEachFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().forEachPrivateName(), forEachFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* hasFunc = JSFunction::create(vm, globalObject, 1, vm.propertyNames->has.string(), setProtoFuncHas, ImplementationVisibility::Public, JSSetHasIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->has, hasFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().hasPrivateName(), hasFunc, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* values = JSFunction::create(vm, globalObject, 0, vm.propertyNames->builtinNames().valuesPublicName().string(), setProtoFuncValues, ImplementationVisibility::Public, JSSetValuesIntrinsic);
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().keysPublicName(), values, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().keysPrivateName(), values, static_cast<unsigned>(PropertyAttribute::DontEnum));

    JSFunction* sizeGetter = JSFunction::create(vm, globalObject, 0, "get size"_s, setProtoFuncSize, ImplementationVisibility::Public);
    GetterSetter* sizeAccessor = GetterSetter::create(vm, globalObject, sizeGetter, nullptr);
    putDirectNonIndexAccessorWithoutTransition(vm, vm.propertyNames->size, sizeAccessor, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);
    putDirectNonIndexAccessorWithoutTransition(vm, vm.propertyNames->builtinNames().sizePrivateName(), sizeAccessor, PropertyAttribute::DontEnum | PropertyAttribute::Accessor);

    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().valuesPublicName(), values, static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->builtinNames().valuesPrivateName(), values, static_cast<unsigned>(PropertyAttribute::DontEnum));

    putDirectWithoutTransition(vm, vm.propertyNames->iteratorSymbol, values, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();

    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().unionPublicName(), setPrototypeUnionCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().intersectionPublicName(), setPrototypeIntersectionCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().differencePublicName(), setPrototypeDifferenceCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().symmetricDifferencePublicName(), setPrototypeSymmetricDifferenceCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().isSubsetOfPublicName(), setPrototypeIsSubsetOfCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().isSupersetOfPublicName(), setPrototypeIsSupersetOfCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().isDisjointFromPublicName(), setPrototypeIsDisjointFromCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));

    globalObject->installSetPrototypeWatchpoint(this);
}

ALWAYS_INLINE static JSSet* getSet(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!thisValue.isCell())) {
        throwVMError(globalObject, scope, createNotAnObjectError(globalObject, thisValue));
        return nullptr;
    }
    auto* set = jsDynamicCast<JSSet*>(thisValue.asCell());
    if (LIKELY(set))
        return set;
    throwTypeError(globalObject, scope, "Set operation called on non-Set object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncAdd, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSSet* set = getSet(globalObject, thisValue);
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));

    set->add(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));
    return JSValue::encode(thisValue);
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncClear, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSSet* set = getSet(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));

    scope.release();
    set->clear(globalObject);
    return JSValue::encode(jsUndefined());
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncDelete, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSSet* set = getSet(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));

    RELEASE_AND_RETURN(scope, JSValue::encode(jsBoolean(set->remove(globalObject, callFrame->argument(0)))));
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncHas, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSSet* set = getSet(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));

    RELEASE_AND_RETURN(scope, JSValue::encode(jsBoolean(set->has(globalObject, callFrame->argument(0)))));
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncSize, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSSet* set = getSet(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, JSValue::encode(jsUndefined()));

    return JSValue::encode(jsNumber(set->size()));
}

inline JSValue createSetIteratorObject(JSGlobalObject* globalObject, CallFrame* callFrame, IterationKind kind)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->thisValue();
    JSSet* set = getSet(globalObject, thisValue);
    RETURN_IF_EXCEPTION(scope, jsUndefined());

    RELEASE_AND_RETURN(scope, JSSetIterator::create(globalObject, globalObject->setIteratorStructure(), set, kind));
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncValues, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(createSetIteratorObject(globalObject, callFrame, IterationKind::Values));
}

JSC_DEFINE_HOST_FUNCTION(setProtoFuncEntries, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(createSetIteratorObject(globalObject, callFrame, IterationKind::Entries));
}

}
