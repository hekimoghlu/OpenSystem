/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

#include "IterationModeMetadata.h"
#include "JSArrayIterator.h"
#include "JSCJSValue.h"
#include "JSObjectInlines.h"
#include "ThrowScope.h"

namespace JSC {

struct IterationRecord {
    JSValue iterator;
    JSValue nextMethod;
};

JSValue iteratorNext(JSGlobalObject*, IterationRecord, JSValue argument = JSValue());
JS_EXPORT_PRIVATE JSValue iteratorValue(JSGlobalObject*, JSValue iterResult);
bool iteratorComplete(JSGlobalObject*, JSValue iterResult);
JS_EXPORT_PRIVATE JSValue iteratorStep(JSGlobalObject*, IterationRecord);
JS_EXPORT_PRIVATE void iteratorClose(JSGlobalObject*, JSValue iterator);
JS_EXPORT_PRIVATE JSObject* createIteratorResultObject(JSGlobalObject*, JSValue, bool done);

Structure* createIteratorResultObjectStructure(VM&, JSGlobalObject&);

JS_EXPORT_PRIVATE JSValue iteratorMethod(JSGlobalObject*, JSObject*);
JS_EXPORT_PRIVATE IterationRecord iteratorForIterable(JSGlobalObject*, JSObject*, JSValue iteratorMethod);
JS_EXPORT_PRIVATE IterationRecord iteratorForIterable(JSGlobalObject*, JSValue iterable);
JS_EXPORT_PRIVATE IterationRecord iteratorDirect(JSGlobalObject*, JSValue);

JS_EXPORT_PRIVATE JSValue iteratorMethod(JSGlobalObject*, JSObject*);
JS_EXPORT_PRIVATE bool hasIteratorMethod(JSGlobalObject*, JSValue);

JS_EXPORT_PRIVATE IterationMode getIterationMode(VM&, JSGlobalObject*, JSValue iterable);
JS_EXPORT_PRIVATE IterationMode getIterationMode(VM&, JSGlobalObject*, JSValue iterable, JSValue symbolIterator);

template<typename CallBackType>
static ALWAYS_INLINE void forEachInFastArray(JSGlobalObject* globalObject, JSValue iterable, JSArray* array, const CallBackType& callback)
{
    UNUSED_PARAM(iterable);

    auto& vm = getVM(globalObject);
    ASSERT(getIterationMode(vm, globalObject, iterable) == IterationMode::FastArray);

    auto scope = DECLARE_THROW_SCOPE(vm);

    for (unsigned index = 0; index < array->length(); ++index) {
        JSValue nextValue = array->getIndex(globalObject, index);
        RETURN_IF_EXCEPTION(scope, void());
        callback(vm, globalObject, nextValue);
        if (UNLIKELY(scope.exception())) {
            scope.release();
            JSArrayIterator* iterator = JSArrayIterator::create(vm, globalObject->arrayIteratorStructure(), array, IterationKind::Values);
            iterator->internalField(JSArrayIterator::Field::Index).setWithoutWriteBarrier(jsNumber(index + 1));
            iteratorClose(globalObject, iterator);
            return;
        }
    }
}

template<typename CallBackType>
static ALWAYS_INLINE void forEachInIterationRecord(JSGlobalObject* globalObject, IterationRecord iterationRecord, const CallBackType& callback)
{
    auto& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    while (true) {
        JSValue next = iteratorStep(globalObject, iterationRecord);
        if (UNLIKELY(scope.exception()) || next.isFalse())
            return;

        JSValue nextValue = iteratorValue(globalObject, next);
        RETURN_IF_EXCEPTION(scope, void());

        callback(vm, globalObject, nextValue);
        if (UNLIKELY(scope.exception())) {
            scope.release();
            iteratorClose(globalObject, iterationRecord.iterator);
            return;
        }
    }
}

template<typename CallBackType>
void forEachInIterable(JSGlobalObject* globalObject, JSValue iterable, const CallBackType& callback)
{
    auto& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (getIterationMode(vm, globalObject, iterable) == IterationMode::FastArray) {
        auto* array = jsCast<JSArray*>(iterable);
        forEachInFastArray(globalObject, iterable, array, callback);
        RETURN_IF_EXCEPTION(scope, void());
        return;
    }

    IterationRecord iterationRecord = iteratorForIterable(globalObject, iterable);
    RETURN_IF_EXCEPTION(scope, void());
    scope.release();
    forEachInIterationRecord(globalObject, iterationRecord, callback);
}

template<typename CallBackType>
void forEachInIterable(JSGlobalObject& globalObject, JSObject* iterable, JSValue iteratorMethod, const CallBackType& callback)
{
    auto& vm = getVM(&globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (getIterationMode(vm, &globalObject, iterable, iteratorMethod) == IterationMode::FastArray) {
        auto* array = jsCast<JSArray*>(iterable);
        for (unsigned index = 0; index < array->length(); ++index) {
            JSValue nextValue = array->getIndex(&globalObject, index);
            RETURN_IF_EXCEPTION(scope, void());
            callback(vm, globalObject, nextValue);
            if (UNLIKELY(scope.exception())) {
                scope.release();
                JSArrayIterator* iterator = JSArrayIterator::create(vm, globalObject.arrayIteratorStructure(), array, IterationKind::Values);
                iterator->internalField(JSArrayIterator::Field::Index).setWithoutWriteBarrier(jsNumber(index + 1));
                iteratorClose(&globalObject, iterator);
                return;
            }
        }
        return;
    }

    auto iterationRecord = iteratorForIterable(&globalObject, iterable, iteratorMethod);
    RETURN_IF_EXCEPTION(scope, void());
    while (true) {
        JSValue next = iteratorStep(&globalObject, iterationRecord);
        if (UNLIKELY(scope.exception()) || next.isFalse())
            return;

        JSValue nextValue = iteratorValue(&globalObject, next);
        RETURN_IF_EXCEPTION(scope, void());

        callback(vm, globalObject, nextValue);
        if (UNLIKELY(scope.exception())) {
            scope.release();
            iteratorClose(&globalObject, iterationRecord.iterator);
            return;
        }
    }
}

template<typename CallBackType>
void forEachInIteratorProtocol(JSGlobalObject* globalObject, JSValue iterable, const CallBackType& callback)
{
    auto& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    IterationRecord iterationRecord = iteratorDirect(globalObject, iterable);
    RETURN_IF_EXCEPTION(scope, void());
    scope.release();
    forEachInIterationRecord(globalObject, iterationRecord, callback);
}

} // namespace JSC
