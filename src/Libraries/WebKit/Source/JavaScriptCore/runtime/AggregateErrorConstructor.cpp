/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include "AggregateErrorConstructor.h"

#include "AggregateError.h"
#include "AggregateErrorPrototype.h"
#include "ClassInfo.h"
#include "ExceptionScope.h"
#include "GCAssertions.h"
#include "JSCInlines.h"
#include "RuntimeType.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(AggregateErrorConstructor);

const ClassInfo AggregateErrorConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(AggregateErrorConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callAggregateErrorConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructAggregateErrorConstructor);

AggregateErrorConstructor::AggregateErrorConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callAggregateErrorConstructor, constructAggregateErrorConstructor)
{
}

void AggregateErrorConstructor::finishCreation(VM& vm, AggregateErrorPrototype* prototype)
{
    Base::finishCreation(vm, 2, errorTypeName(ErrorType::AggregateError), PropertyAdditionMode::WithoutStructureTransition);
    ASSERT(inherits(info()));

    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
}

JSC_DEFINE_HOST_FUNCTION(callAggregateErrorConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    JSValue errors = callFrame->argument(0);
    JSValue message = callFrame->argument(1);
    JSValue options = callFrame->argument(2);
    Structure* errorStructure = globalObject->errorStructure(ErrorType::AggregateError);
    return JSValue::encode(createAggregateError(globalObject, vm, errorStructure, errors, message, options, nullptr, TypeNothing, false));
}

JSC_DEFINE_HOST_FUNCTION(constructAggregateErrorConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue errors = callFrame->argument(0);
    JSValue message = callFrame->argument(1);
    JSValue options = callFrame->argument(2);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* errorStructure = JSC_GET_DERIVED_STRUCTURE(vm, errorStructureWithErrorType<ErrorType::AggregateError>, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });
    ASSERT(errorStructure);

    RELEASE_AND_RETURN(scope, JSValue::encode(createAggregateError(globalObject, vm, errorStructure, errors, message, options, nullptr, TypeNothing, false)));
}

} // namespace JSC
