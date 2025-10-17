/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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
#ifndef APIUtils_h
#define APIUtils_h

#include "CatchScope.h"
#include "Exception.h"
#include "JSCJSValue.h"
#include "JSGlobalObjectInspectorController.h"
#include "JSValueRef.h"

enum class ExceptionStatus {
    DidThrow,
    DidNotThrow
};

inline ExceptionStatus handleExceptionIfNeeded(JSC::CatchScope& scope, JSContextRef ctx, JSValueRef* returnedExceptionRef)
{
    JSC::JSGlobalObject* globalObject = toJS(ctx);
    if (UNLIKELY(scope.exception())) {
        JSC::Exception* exception = scope.exception();
        if (returnedExceptionRef)
            *returnedExceptionRef = toRef(globalObject, exception->value());
        scope.clearException();
#if ENABLE(REMOTE_INSPECTOR)
        globalObject->inspectorController().reportAPIException(globalObject, exception);
#endif
        return ExceptionStatus::DidThrow;
    }
    return ExceptionStatus::DidNotThrow;
}

inline void setException(JSContextRef ctx, JSValueRef* returnedExceptionRef, JSC::JSValue exception)
{
    JSC::JSGlobalObject* globalObject = toJS(ctx);
    if (returnedExceptionRef)
        *returnedExceptionRef = toRef(globalObject, exception);
#if ENABLE(REMOTE_INSPECTOR)
    JSC::VM& vm = getVM(globalObject);
    globalObject->inspectorController().reportAPIException(globalObject, JSC::Exception::create(vm, exception));
#endif
}

#endif /* APIUtils_h */
