/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "DeferTermination.h"
#include "ErrorInstance.h"
#include "Exception.h"
#include "JSObject.h"
#include "ThrowScope.h"

namespace JSC {

typedef JSObject* (*ErrorFactory)(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);

String defaultSourceAppender(const String&, StringView, RuntimeType, ErrorInstance::SourceTextWhereErrorOccurred);
String notAFunctionSourceAppender(const String&, StringView, RuntimeType, ErrorInstance::SourceTextWhereErrorOccurred);

String constructErrorMessage(JSGlobalObject*, JSValue, const String&);
JS_EXPORT_PRIVATE JSObject* createError(JSGlobalObject*, JSValue, const String&, ErrorInstance::SourceAppender);
JS_EXPORT_PRIVATE JSObject* createStackOverflowError(JSGlobalObject*);
JSObject* createUndefinedVariableError(JSGlobalObject*, const Identifier&);
JSObject* createTDZError(JSGlobalObject*);
JSObject* createNotAnObjectError(JSGlobalObject*, JSValue);
JSObject* createInvalidFunctionApplyParameterError(JSGlobalObject*, JSValue);
JSObject* createInvalidInParameterError(JSGlobalObject*, JSValue);
JSObject* createInvalidInstanceofParameterErrorNotFunction(JSGlobalObject*, JSValue);
JSObject* createInvalidInstanceofParameterErrorHasInstanceValueNotFunction(JSGlobalObject*, JSValue);
JSObject* createNotAConstructorError(JSGlobalObject*, JSValue);
JSObject* createNotAFunctionError(JSGlobalObject*, JSValue);
JSObject* createInvalidPrototypeError(JSGlobalObject*, JSValue);
JSObject* createErrorForDuplicateGlobalVariableDeclaration(JSGlobalObject*, UniquedStringImpl*);
JSObject* createErrorForInvalidGlobalFunctionDeclaration(JSGlobalObject*, const Identifier&);
JSObject* createErrorForInvalidGlobalVarDeclaration(JSGlobalObject*, const Identifier&);
JSObject* createInvalidPrivateNameError(JSGlobalObject*);
JSObject* createRedefinedPrivateNameError(JSGlobalObject*);
String errorDescriptionForValue(JSGlobalObject*, JSValue);

JS_EXPORT_PRIVATE Exception* throwOutOfMemoryError(JSGlobalObject*, ThrowScope&);
JS_EXPORT_PRIVATE Exception* throwOutOfMemoryError(JSGlobalObject*, ThrowScope&, const String&);
JS_EXPORT_PRIVATE Exception* throwStackOverflowError(JSGlobalObject*, ThrowScope&);

#if ASSERT_ENABLED

#define DEFER_TERMINATION_AND_ASSERT(vm, assertion, ...) do { \
        JSC::DeferTerminationForAWhile deferScope(vm); \
        ASSERT(assertion, __VA_ARGS__); \
    } while (false)

#define DEFER_TERMINATION_AND_ASSERT_WITH_MESSAGE(vm, assertion, ...) do { \
        JSC::DeferTerminationForAWhile deferScope(vm); \
        ASSERT_WITH_MESSAGE(assertion, __VA_ARGS__); \
    } while (false)

#else

#define DEFER_TERMINATION_AND_ASSERT(vm, assertion, ...) UNUSED_PARAM(vm)
#define DEFER_TERMINATION_AND_ASSERT_WITH_MESSAGE(vm, assertion, ...) UNUSED_PARAM(vm)

#endif // ASSERT_ENABLED

#define DEFER_TERMINATION_AND_RELEASE_ASSERT(vm, assertion, ...) do { \
        JSC::DeferTerminationForAWhile deferScope(vm); \
        RELEASE_ASSERT(assertion, __VA_ARGS__); \
    } while (false)

#define DEFER_TERMINATION_AND_RELEASE_ASSERT_WITH_MESSAGE(vm, assertion, ...) do { \
        JSC::DeferTerminationForAWhile deferScope(vm); \
        RELEASE_ASSERT_WITH_MESSAGE(assertion, __VA_ARGS__); \
    } while (false)

} // namespace JSC
