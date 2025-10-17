/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "Operations.h"

#include "JSBigInt.h"
#include "JSCInlines.h"

namespace JSC {

bool JSValue::equalSlowCase(JSGlobalObject* globalObject, JSValue v1, JSValue v2)
{
    return equalSlowCaseInline(globalObject, v1, v2);
}

NEVER_INLINE JSValue jsAddSlowCase(JSGlobalObject* globalObject, JSValue v1, JSValue v2)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue p1 = v1.toPrimitive(globalObject);
    RETURN_IF_EXCEPTION(scope, { });
    JSValue p2 = v2.toPrimitive(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    if (p1.isString()) {
        if (p2.isCell()) {
            JSString* p2String = p2.toString(globalObject);
            RETURN_IF_EXCEPTION(scope, { });
            RELEASE_AND_RETURN(scope, jsString(globalObject, asString(p1), p2String));
        }
        String p2String = p2.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        RELEASE_AND_RETURN(scope, jsString(globalObject, asString(p1), p2String));
    }

    if (p2.isString()) {
        if (p1.isCell()) {
            JSString* p1String = p1.toString(globalObject);
            RETURN_IF_EXCEPTION(scope, { });
            RELEASE_AND_RETURN(scope, jsString(globalObject, p1String, asString(p2)));
        }
        String p1String = p1.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(scope, { });
        RELEASE_AND_RETURN(scope, jsString(globalObject, p1String, asString(p2)));
    }

    auto doubleOp = [] (double left, double right) -> double {
        return left + right;
    };

    auto bigIntOp = [] (JSGlobalObject* globalObject, auto left, auto right) {
        return JSBigInt::add(globalObject, left, right);
    };

    RELEASE_AND_RETURN(scope, arithmeticBinaryOp(globalObject, p1, p2, doubleOp, bigIntOp, "Invalid mix of BigInt and other type in addition."_s));
}

JSString* jsTypeStringForValueWithConcurrency(VM& vm, JSGlobalObject* globalObject, JSValue v, Concurrency concurrency)
{
    if (v.isUndefined())
        return vm.smallStrings.undefinedString();
    if (v.isBoolean())
        return vm.smallStrings.booleanString();
    if (v.isNumber())
        return vm.smallStrings.numberString();
    if (v.isString())
        return vm.smallStrings.stringString();
    if (v.isSymbol())
        return vm.smallStrings.symbolString();
    if (v.isBigInt())
        return vm.smallStrings.bigintString();
    if (v.isObject()) {
        JSObject* object = asObject(v);
        // Return "undefined" for objects that should be treated
        // as null when doing comparisons.
        if (object->structure()->masqueradesAsUndefined(globalObject))
            return vm.smallStrings.undefinedString();
        if (LIKELY(concurrency == Concurrency::MainThread)) {
            if (object->isCallable())
                return vm.smallStrings.functionString();
            return vm.smallStrings.objectString();
        }

        switch (object->isCallableWithConcurrency<Concurrency::ConcurrentThread>()) {
        case TriState::True:
            return vm.smallStrings.functionString();
        case TriState::False:
            return vm.smallStrings.objectString();
        case TriState::Indeterminate:
            return nullptr;
        }
    }
    // This case can happen for internal objects like GetterSetter, that
    // can be exposed to this function by transformations like LICM blind hoisting.
    // The actual result shouldn't matter, as long as it matches buildTypeOf.
    return vm.smallStrings.objectString();
}

size_t normalizePrototypeChain(JSGlobalObject* globalObject, JSCell* base, bool& sawPolyProto)
{
    VM& vm = globalObject->vm();
    size_t count = 0;
    sawPolyProto = false;
    JSCell* current = base;
    while (1) {
        Structure* structure = current->structure();
        if (structure->isProxy())
            return InvalidPrototypeChain;

        sawPolyProto |= structure->hasPolyProto();

        JSValue prototype = structure->prototypeForLookup(globalObject, current);
        if (prototype.isNull())
            return count;

        current = prototype.asCell();
        structure = current->structure();
        if (structure->isDictionary()) {
            if (structure->hasBeenFlattenedBefore())
                return InvalidPrototypeChain;
            structure->flattenDictionaryStructure(vm, asObject(current));
        }

        ++count;
    }
}

} // namespace JSC
