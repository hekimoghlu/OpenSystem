/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "IteratorOperations.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"

namespace JSC {

JSValue iteratorNext(JSGlobalObject* globalObject, IterationRecord iterationRecord, JSValue argument)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue iterator = iterationRecord.iterator;
    JSValue nextFunction = iterationRecord.nextMethod;

    auto nextFunctionCallData = JSC::getCallData(nextFunction);
    if (nextFunctionCallData.type == CallData::Type::None)
        return throwTypeError(globalObject, scope);

    MarkedArgumentBuffer nextFunctionArguments;
    if (!argument.isEmpty())
        nextFunctionArguments.append(argument);
    ASSERT(!nextFunctionArguments.hasOverflowed());
    JSValue result = call(globalObject, nextFunction, nextFunctionCallData, iterator, nextFunctionArguments);
    RETURN_IF_EXCEPTION(scope, JSValue());

    if (!result.isObject())
        return throwTypeError(globalObject, scope, "Iterator result interface is not an object."_s);

    return result;
}

JSValue iteratorValue(JSGlobalObject* globalObject, JSValue iterResult)
{
    return iterResult.get(globalObject, globalObject->vm().propertyNames->value);
}

bool iteratorComplete(JSGlobalObject* globalObject, JSValue iterResult)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue done = iterResult.get(globalObject, globalObject->vm().propertyNames->done);
    RETURN_IF_EXCEPTION(scope, true);
    RELEASE_AND_RETURN(scope, done.toBoolean(globalObject));
}

JSValue iteratorStep(JSGlobalObject* globalObject, IterationRecord iterationRecord)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue result = iteratorNext(globalObject, iterationRecord);
    RETURN_IF_EXCEPTION(scope, JSValue());
    bool done = iteratorComplete(globalObject, result);
    RETURN_IF_EXCEPTION(scope, JSValue());
    if (done)
        return jsBoolean(false);
    return result;
}

void iteratorClose(JSGlobalObject* globalObject, JSValue iterator)
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto catchScope = DECLARE_CATCH_SCOPE(vm);

    Exception* exception = nullptr;
    if (UNLIKELY(catchScope.exception())) {
        exception = catchScope.exception();
        catchScope.clearException();
    }

    JSValue returnFunction = iterator.get(globalObject, vm.propertyNames->returnKeyword);
    if (UNLIKELY(throwScope.exception()) || returnFunction.isUndefinedOrNull()) {
        if (exception)
            throwException(globalObject, throwScope, exception);
        return;
    }

    auto returnFunctionCallData = JSC::getCallData(returnFunction);
    if (returnFunctionCallData.type == CallData::Type::None) {
        if (exception)
            throwException(globalObject, throwScope, exception);
        else
            throwTypeError(globalObject, throwScope);
        return;
    }

    MarkedArgumentBuffer returnFunctionArguments;
    ASSERT(!returnFunctionArguments.hasOverflowed());
    JSValue innerResult = call(globalObject, returnFunction, returnFunctionCallData, iterator, returnFunctionArguments);

    if (exception) {
        throwException(globalObject, throwScope, exception);
        return;
    }

    RETURN_IF_EXCEPTION(throwScope, void());

    if (!innerResult.isObject()) {
        throwTypeError(globalObject, throwScope, "Iterator result interface is not an object."_s);
        return;
    }
}

static constexpr PropertyOffset valuePropertyOffset = 0;
static constexpr PropertyOffset donePropertyOffset = 1;

Structure* createIteratorResultObjectStructure(VM& vm, JSGlobalObject& globalObject)
{
    Structure* iteratorResultStructure = globalObject.structureCache().emptyObjectStructureForPrototype(&globalObject, globalObject.objectPrototype(), JSFinalObject::defaultInlineCapacity);
    PropertyOffset offset;
    iteratorResultStructure = Structure::addPropertyTransition(vm, iteratorResultStructure, vm.propertyNames->value, 0, offset);
    RELEASE_ASSERT(offset == valuePropertyOffset);
    iteratorResultStructure = Structure::addPropertyTransition(vm, iteratorResultStructure, vm.propertyNames->done, 0, offset);
    RELEASE_ASSERT(offset == donePropertyOffset);
    return iteratorResultStructure;
}

JSObject* createIteratorResultObject(JSGlobalObject* globalObject, JSValue value, bool done)
{
    VM& vm = globalObject->vm();
    JSObject* resultObject = constructEmptyObject(vm, globalObject->iteratorResultObjectStructure());
    resultObject->putDirectOffset(vm, valuePropertyOffset, value);
    resultObject->putDirectOffset(vm, donePropertyOffset, jsBoolean(done));
    return resultObject;
}

bool hasIteratorMethod(JSGlobalObject* globalObject, JSValue value)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!value.isObject())
        return false;

    JSObject* object = asObject(value);
    CallData callData;
    JSValue applyMethod = object->getMethod(globalObject, callData, vm.propertyNames->iteratorSymbol, "Symbol.iterator property should be callable"_s);
    RETURN_IF_EXCEPTION(scope, false);

    return !applyMethod.isUndefined();
}

JSValue iteratorMethod(JSGlobalObject* globalObject, JSObject* object)
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    CallData callData;
    JSValue method = object->getMethod(globalObject, callData, vm.propertyNames->iteratorSymbol, "Symbol.iterator property should be callable"_s);
    RETURN_IF_EXCEPTION(scope, jsUndefined());

    return method;
}

IterationRecord iteratorForIterable(JSGlobalObject* globalObject, JSObject* object, JSValue iteratorMethod)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto iteratorMethodCallData = JSC::getCallData(iteratorMethod);
    if (iteratorMethodCallData.type == CallData::Type::None) {
        throwTypeError(globalObject, scope);
        return { };
    }

    ArgList iteratorMethodArguments;
    JSValue iterator = call(globalObject, iteratorMethod, iteratorMethodCallData, object, iteratorMethodArguments);
    RETURN_IF_EXCEPTION(scope, { });

    if (!iterator.isObject()) {
        throwTypeError(globalObject, scope);
        return { };
    }

    JSValue nextMethod = iterator.getObject()->get(globalObject, vm.propertyNames->next);
    RETURN_IF_EXCEPTION(scope, { });

    return { iterator, nextMethod };
}

IterationRecord iteratorForIterable(JSGlobalObject* globalObject, JSValue iterable)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    
    JSValue iteratorFunction = iterable.get(globalObject, vm.propertyNames->iteratorSymbol);
    RETURN_IF_EXCEPTION(scope, { });
    
    auto iteratorFunctionCallData = JSC::getCallData(iteratorFunction);
    if (iteratorFunctionCallData.type == CallData::Type::None) {
        throwTypeError(globalObject, scope);
        return { };
    }

    ArgList iteratorFunctionArguments;
    JSValue iterator = call(globalObject, iteratorFunction, iteratorFunctionCallData, iterable, iteratorFunctionArguments);
    RETURN_IF_EXCEPTION(scope, { });

    if (!iterator.isObject()) {
        throwTypeError(globalObject, scope);
        return { };
    }

    JSValue nextMethod = iterator.getObject()->get(globalObject, vm.propertyNames->next);
    RETURN_IF_EXCEPTION(scope, { });

    return { iterator, nextMethod };
}

IterationRecord iteratorDirect(JSGlobalObject* globalObject, JSValue object)
{
    return { object, object.get(globalObject, globalObject->vm().propertyNames->next) };
}

IterationMode getIterationMode(VM&, JSGlobalObject* globalObject, JSValue iterable, JSValue symbolIterator)
{
    if (!isJSArray(iterable))
        return IterationMode::Generic;

    if (!globalObject->arrayIteratorProtocolWatchpointSet().isStillValid())
        return IterationMode::Generic;

    // This is correct because we just checked the watchpoint is still valid.
    JSFunction* symbolIteratorFunction = jsDynamicCast<JSFunction*>(symbolIterator);
    if (!symbolIteratorFunction)
        return IterationMode::Generic;

    // We don't want to allocate the values function just to check if it's the same as our function so we use the concurrent accessor.
    // FIXME: This only works for arrays from the same global object as ourselves but we should be able to support any pairing.
    if (globalObject->arrayProtoValuesFunctionConcurrently() != symbolIteratorFunction)
        return IterationMode::Generic;

    return IterationMode::FastArray;
}

IterationMode getIterationMode(VM&, JSGlobalObject* globalObject, JSValue iterable)
{
    if (!isJSArray(iterable))
        return IterationMode::Generic;

    JSArray* array = jsCast<JSArray*>(iterable);
    Structure* structure = array->structure();
    // FIXME: We want to support broader JSArrays as long as array[@@iterator] is not defined.
    if (!globalObject->isOriginalArrayStructure(structure))
        return IterationMode::Generic;

    if (!globalObject->arrayIteratorProtocolWatchpointSet().isStillValid())
        return IterationMode::Generic;

    // Now, Array has original Array Structures and arrayIteratorProtocolWatchpointSet is not fired.
    // This means,
    // 1. Array.prototype is [[Prototype]].
    // 2. array[@@iterator] is not overridden.
    // 3. Array.prototype[@@iterator] is an expected one.
    // So, we can say this will create an expected ArrayIterator.
    return IterationMode::FastArray;
}

} // namespace JSC
