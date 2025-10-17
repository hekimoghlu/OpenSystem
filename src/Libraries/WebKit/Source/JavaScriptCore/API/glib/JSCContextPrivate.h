/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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

#include "JSCContext.h"
#include "JSCValue.h"
#include <wtf/glib/GRefPtr.h>

namespace JSC {
class JSObject;
}

typedef struct OpaqueJSClass* JSClassRef;
typedef struct OpaqueJSContext* JSGlobalContextRef;
typedef const struct OpaqueJSValue* JSValueRef;
typedef struct OpaqueJSValue* JSObjectRef;

JS_EXPORT_PRIVATE GRefPtr<JSCContext> jscContextGetOrCreate(JSGlobalContextRef);
JS_EXPORT_PRIVATE JSGlobalContextRef jscContextGetJSContext(JSCContext*);
JS_EXPORT_PRIVATE GRefPtr<JSCValue> jscContextGetOrCreateValue(JSCContext*, JSValueRef);
void jscContextValueDestroyed(JSCContext*, JSValueRef);
JSC::JSObject* jscContextGetJSWrapper(JSCContext*, gpointer);
JSC::JSObject* jscContextGetOrCreateJSWrapper(JSCContext*, JSClassRef, JSValueRef prototype = nullptr, gpointer = nullptr, GDestroyNotify = nullptr);
JSGlobalContextRef jscContextCreateContextWithJSWrapper(JSCContext*, JSClassRef, JSValueRef prototype = nullptr, gpointer = nullptr, GDestroyNotify = nullptr);
gpointer jscContextWrappedObject(JSCContext*, JSObjectRef);
JSCClass* jscContextGetRegisteredClass(JSCContext*, JSClassRef);

struct CallbackData {
    GRefPtr<JSCContext> context;
    GRefPtr<JSCException> preservedException;
    JSValueRef calleeValue;
    JSValueRef thisValue;
    size_t argumentCount;
    const JSValueRef* arguments;

    CallbackData* next;
};
CallbackData jscContextPushCallback(JSCContext*, JSValueRef calleeValue, JSValueRef thisValue, size_t argumentCount, const JSValueRef* arguments);
void jscContextPopCallback(JSCContext*, CallbackData&&);

bool jscContextHandleExceptionIfNeeded(JSCContext*, JSValueRef);
JSValueRef jscContextGArrayToJSArray(JSCContext*, GPtrArray*, JSValueRef* exception);
JSValueRef jscContextGValueToJSValue(JSCContext*, const GValue*, JSValueRef* exception);
void jscContextJSValueToGValue(JSCContext*, JSValueRef, GType, GValue*, JSValueRef* exception);
