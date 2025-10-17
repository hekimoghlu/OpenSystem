/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
#include "AggregateError.h"

#include "ExceptionScope.h"
#include "IteratorOperations.h"
#include "JSCJSValueInlines.h"
#include "JSGlobalObjectInlines.h"

namespace JSC {

ErrorInstance* createAggregateError(JSGlobalObject* globalObject, VM& vm, Structure* structure, JSValue errors, JSValue message, JSValue options, ErrorInstance::SourceAppender appender, RuntimeType type, bool useCurrentFrame)
{
    auto scope = DECLARE_THROW_SCOPE(vm);

    String messageString = message.isUndefined() ? String() : message.toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, nullptr);

    JSValue cause;
    if (options.isObject()) {
        // Since `throw undefined;` is valid, we need to distinguish the case where `cause` is an explicit undefined.
        cause = asObject(options)->getIfPropertyExists(globalObject, vm.propertyNames->cause);
        RETURN_IF_EXCEPTION(scope, nullptr);
    }

    MarkedArgumentBuffer errorsList;
    forEachInIterable(globalObject, errors, [&] (VM&, JSGlobalObject*, JSValue nextValue) {
        errorsList.append(nextValue);
        if (UNLIKELY(errorsList.hasOverflowed()))
            throwOutOfMemoryError(globalObject, scope);
    });
    RETURN_IF_EXCEPTION(scope, nullptr);

    auto* array = constructArray(globalObject, static_cast<ArrayAllocationProfile*>(nullptr), errorsList);
    RETURN_IF_EXCEPTION(scope, nullptr);

    auto* error = ErrorInstance::create(vm, structure, messageString, cause, appender, type, ErrorType::AggregateError, useCurrentFrame);
    error->putDirect(vm, vm.propertyNames->errors, array, static_cast<unsigned>(PropertyAttribute::DontEnum));
    return error;
}

} // namespace JSC
