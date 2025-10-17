/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "WebKitJavascriptResult.h"

#if !ENABLE(2022_GLIB_API)

#include "APISerializedScriptValue.h"
#include "WebKitJavascriptResultPrivate.h"
#include <jsc/JSCContextPrivate.h>
#include <jsc/JSCValuePrivate.h>

/**
 * WebKitJavascriptResult: (ref-func webkit_javascript_result_ref) (unref-func webkit_javascript_result_unref)
 *
 * Result of JavaScript evaluation in a web view.
 */

struct _WebKitJavascriptResult {
    explicit _WebKitJavascriptResult(WebCore::SerializedScriptValue& serializedScriptValue)
    {
        jsValue = API::SerializedScriptValue::deserialize(serializedScriptValue);
    }

    GRefPtr<JSCValue> jsValue;

    int referenceCount { 1 };
};

G_DEFINE_BOXED_TYPE(WebKitJavascriptResult, webkit_javascript_result, webkit_javascript_result_ref, webkit_javascript_result_unref)

WebKitJavascriptResult* webkitJavascriptResultCreate(WebCore::SerializedScriptValue& serializedScriptValue)
{
    WebKitJavascriptResult* result = static_cast<WebKitJavascriptResult*>(fastMalloc(sizeof(WebKitJavascriptResult)));
    new (result) WebKitJavascriptResult(serializedScriptValue);
    return result;
}

/**
 * webkit_javascript_result_ref:
 * @js_result: a #WebKitJavascriptResult
 *
 * Atomically increments the reference count of @js_result by one.
 *
 * This function is MT-safe and may be called from any thread.
 *
 * Returns: The passed in #WebKitJavascriptResult
 */
WebKitJavascriptResult* webkit_javascript_result_ref(WebKitJavascriptResult* javascriptResult)
{
    g_atomic_int_inc(&javascriptResult->referenceCount);
    return javascriptResult;
}

/**
 * webkit_javascript_result_unref:
 * @js_result: a #WebKitJavascriptResult
 *
 * Atomically decrements the reference count of @js_result by one.
 *
 * If the reference count drops to 0,
 * all memory allocated by the #WebKitJavascriptResult is
 * released. This function is MT-safe and may be called from any
 * thread.
 */
void webkit_javascript_result_unref(WebKitJavascriptResult* javascriptResult)
{
    if (g_atomic_int_dec_and_test(&javascriptResult->referenceCount)) {
        javascriptResult->~WebKitJavascriptResult();
        fastFree(javascriptResult);
    }
}

#if PLATFORM(GTK)
/**
 * webkit_javascript_result_get_global_context: (skip)
 * @js_result: a #WebKitJavascriptResult
 *
 * Get the global Javascript context.
 *
 * Get the global Javascript context that should be used with the
 * <function>JSValueRef</function> returned by webkit_javascript_result_get_value().
 *
 * Returns: the <function>JSGlobalContextRef</function> for the #WebKitJavascriptResult
 *
 * Deprecated: 2.22: Use jsc_value_get_context() instead.
 */
JSGlobalContextRef webkit_javascript_result_get_global_context(WebKitJavascriptResult* javascriptResult)
{
    g_return_val_if_fail(javascriptResult, nullptr);
    return jscContextGetJSContext(jsc_value_get_context(javascriptResult->jsValue.get()));
}

/**
 * webkit_javascript_result_get_value: (skip)
 * @js_result: a #WebKitJavascriptResult
 *
 * Get the value of @js_result.
 *
 * You should use the <function>JSGlobalContextRef</function>
 * returned by webkit_javascript_result_get_global_context() to use the <function>JSValueRef</function>.
 *
 * Returns: the <function>JSValueRef</function> of the #WebKitJavascriptResult
 *
 * Deprecated: 2.22: Use webkit_javascript_result_get_js_value() instead.
 */
JSValueRef webkit_javascript_result_get_value(WebKitJavascriptResult* javascriptResult)
{
    g_return_val_if_fail(javascriptResult, nullptr);
    return jscValueGetJSValue(javascriptResult->jsValue.get());
}
#endif

/**
 * webkit_javascript_result_get_js_value:
 * @js_result: a #WebKitJavascriptResult
 *
 * Get the #JSCValue of @js_result.
 *
 * Returns: (transfer none): the #JSCValue of the #WebKitJavascriptResult
 *
 * Since: 2.22
 */
JSCValue* webkit_javascript_result_get_js_value(WebKitJavascriptResult* javascriptResult)
{
    g_return_val_if_fail(javascriptResult, nullptr);
    return javascriptResult->jsValue.get();
}

#endif
