/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "BigIntPrototype.h"

#include "BigIntObject.h"
#include "IntegrityInlines.h"
#include "IntlNumberFormatInlines.h"
#include "JSBigInt.h"
#include "JSCInlines.h"
#include "JSCast.h"
#include "NumberPrototype.h"
#include <wtf/Assertions.h>

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(bigIntProtoFuncToString);
static JSC_DECLARE_HOST_FUNCTION(bigIntProtoFuncToLocaleString);
static JSC_DECLARE_HOST_FUNCTION(bigIntProtoFuncValueOf);

}

#include "BigIntPrototype.lut.h"

namespace JSC {

const ClassInfo BigIntPrototype::s_info = { "BigInt"_s, &Base::s_info, &bigIntPrototypeTable, nullptr, CREATE_METHOD_TABLE(BigIntPrototype) };

/* Source for BigIntPrototype.lut.h
@begin bigIntPrototypeTable
  toString          bigIntProtoFuncToString         DontEnum|Function 0
  toLocaleString    bigIntProtoFuncToLocaleString   DontEnum|Function 0
  valueOf           bigIntProtoFuncValueOf          DontEnum|Function 0
@end
*/

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(BigIntPrototype);

BigIntPrototype::BigIntPrototype(VM& vm, Structure* structure)
    : JSNonFinalObject(vm, structure)
{
}

void BigIntPrototype::finishCreation(VM& vm, JSGlobalObject*)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// ------------------------------ Functions ---------------------------

static ALWAYS_INLINE JSBigInt* toThisBigIntValue(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

#if USE(BIGINT32)
    // FIXME: heap-allocating all big ints is inneficient, but re-implementing toString for small BigInts is enough work that I'm deferring it to a later patch.
    if (thisValue.isBigInt32())
        RELEASE_AND_RETURN(scope, JSBigInt::createFrom(globalObject, thisValue.bigInt32AsInt32()));
#endif

    if (thisValue.isCell()) {
        if (JSBigInt* bigInt = jsDynamicCast<JSBigInt*>(thisValue.asCell()))
            return bigInt;

        if (BigIntObject* bigIntObject = jsDynamicCast<BigIntObject*>(thisValue.asCell())) {
            JSValue bigInt = bigIntObject->internalValue();
#if USE(BIGINT32)
            if (bigInt.isBigInt32())
                RELEASE_AND_RETURN(scope, JSBigInt::createFrom(globalObject, bigInt.bigInt32AsInt32()));
#endif
            return bigInt.asHeapBigInt();
        }
    }

    throwTypeError(globalObject, scope, "'this' value must be a BigInt or BigIntObject"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(bigIntProtoFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSBigInt* value = toThisBigIntValue(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });

    ASSERT(value);

    Integrity::auditStructureID(value->structureID());
    int32_t radix = extractToStringRadixArgument(globalObject, callFrame->argument(0), scope);
    RETURN_IF_EXCEPTION(scope, { });

    String resultString = value->toString(globalObject, radix);
    RETURN_IF_EXCEPTION(scope, { });
    scope.release();
    if (resultString.length() == 1)
        return JSValue::encode(vm.smallStrings.singleCharacterString(resultString[0]));

    return JSValue::encode(jsNontrivialString(vm, resultString));
}

// FIXME: this function should introduce the right separators for thousands and similar things.
JSC_DEFINE_HOST_FUNCTION(bigIntProtoFuncToLocaleString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSBigInt* thisValue = toThisBigIntValue(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });

    auto* numberFormat = IntlNumberFormat::create(vm, globalObject->numberFormatStructure());
    numberFormat->initializeNumberFormat(globalObject, callFrame->argument(0), callFrame->argument(1));
    RETURN_IF_EXCEPTION(scope, { });

    auto value = toIntlMathematicalValue(globalObject, thisValue);
    RETURN_IF_EXCEPTION(scope, { });

    if (auto number = value.tryGetDouble())
        RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->format(globalObject, number.value())));

    RELEASE_AND_RETURN(scope, JSValue::encode(numberFormat->format(globalObject, WTFMove(value))));
}

JSC_DEFINE_HOST_FUNCTION(bigIntProtoFuncValueOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSBigInt* value = toThisBigIntValue(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });

    Integrity::auditStructureID(value->structureID());
    return JSValue::encode(value);
}

} // namespace JSC
