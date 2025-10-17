/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "BooleanPrototype.h"

#include "IntegrityInlines.h"
#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(booleanProtoFuncToString);
static JSC_DECLARE_HOST_FUNCTION(booleanProtoFuncValueOf);

}

#include "BooleanPrototype.lut.h"

namespace JSC {

const ClassInfo BooleanPrototype::s_info = { "Boolean"_s, &BooleanObject::s_info, &booleanPrototypeTable, nullptr, CREATE_METHOD_TABLE(BooleanPrototype) };

/* Source for BooleanPrototype.lut.h
@begin booleanPrototypeTable
  toString  booleanProtoFuncToString    DontEnum|Function 0
  valueOf   booleanProtoFuncValueOf     DontEnum|Function 0
@end
*/

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(BooleanPrototype);

BooleanPrototype::BooleanPrototype(VM& vm, Structure* structure)
    : BooleanObject(vm, structure)
{
}

void BooleanPrototype::finishCreation(VM& vm, JSGlobalObject*)
{
    Base::finishCreation(vm);
    setInternalValue(vm, jsBoolean(false));

    ASSERT(inherits(info()));
}

// ------------------------------ Functions ---------------------------

JSC_DEFINE_HOST_FUNCTION(booleanProtoFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue thisValue = callFrame->thisValue();
    if (thisValue == jsBoolean(false))
        return JSValue::encode(vm.smallStrings.falseString());

    if (thisValue == jsBoolean(true))
        return JSValue::encode(vm.smallStrings.trueString());

    auto* thisObject = jsDynamicCast<BooleanObject*>(thisValue);
    if (UNLIKELY(!thisObject))
        return throwVMTypeError(globalObject, scope);

    Integrity::auditStructureID(thisObject->structureID());
    if (thisObject->internalValue() == jsBoolean(false))
        return JSValue::encode(vm.smallStrings.falseString());

    ASSERT(thisObject->internalValue() == jsBoolean(true));
    return JSValue::encode(vm.smallStrings.trueString());
}

JSC_DEFINE_HOST_FUNCTION(booleanProtoFuncValueOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue thisValue = callFrame->thisValue();
    if (thisValue.isBoolean())
        return JSValue::encode(thisValue);

    auto* thisObject = jsDynamicCast<BooleanObject*>(thisValue);
    if (UNLIKELY(!thisObject))
        return throwVMTypeError(globalObject, scope);

    Integrity::auditStructureID(thisObject->structureID());
    return JSValue::encode(thisObject->internalValue());
}

} // namespace JSC
