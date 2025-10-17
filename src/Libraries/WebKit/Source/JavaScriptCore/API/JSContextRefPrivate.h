/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#ifndef JSContextRefPrivate_h
#define JSContextRefPrivate_h

#include <JavaScriptCore/JSObjectRef.h>
#include <JavaScriptCore/JSValueRef.h>
#include <JavaScriptCore/WebKitAvailability.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*!
@function
@abstract Gets a Backtrace for the existing context
@param ctx The JSContext whose backtrace you want to get
@result A string containing the backtrace
*/
JS_EXPORT JSStringRef JSContextCreateBacktrace(JSContextRef ctx, unsigned maxStackSize) JSC_API_AVAILABLE(macos(10.6), ios(7.0));
    

/*! 
@typedef JSShouldTerminateCallback
@abstract The callback invoked when script execution has exceeded the allowed
 time limit previously specified via JSContextGroupSetExecutionTimeLimit.
@param ctx The execution context to use.
@param context User specified context data previously passed to
 JSContextGroupSetExecutionTimeLimit.
@discussion If you named your function Callback, you would declare it like this:

 bool Callback(JSContextRef ctx, void* context);

 If you return true, the timed out script will terminate.
 If you return false, the script will run for another period of the allowed
 time limit specified via JSContextGroupSetExecutionTimeLimit.

 Within this callback function, you may call JSContextGroupSetExecutionTimeLimit
 to set a new time limit, or JSContextGroupClearExecutionTimeLimit to cancel the
 timeout.
*/
typedef bool
(*JSShouldTerminateCallback) (JSContextRef ctx, void* context);

/*!
@function
@abstract Sets the script execution time limit.
@param group The JavaScript context group that this time limit applies to.
@param limit The time limit of allowed script execution time in seconds.
@param callback The callback function that will be invoked when the time limit
 has been reached. This will give you a chance to decide if you want to
 terminate the script or not. If you pass a NULL callback, the script will be
 terminated unconditionally when the time limit has been reached.
@param context User data that you can provide to be passed back to you
 in your callback.

 In order to guarantee that the execution time limit will take effect, you will
 need to call JSContextGroupSetExecutionTimeLimit before you start executing
 any scripts.
*/
JS_EXPORT void JSContextGroupSetExecutionTimeLimit(JSContextGroupRef group, double limit, JSShouldTerminateCallback callback, void* context) JSC_API_AVAILABLE(macos(10.6), ios(7.0));

/*!
@function
@abstract Clears the script execution time limit.
@param group The JavaScript context group that the time limit is cleared on.
*/
JS_EXPORT void JSContextGroupClearExecutionTimeLimit(JSContextGroupRef group) JSC_API_AVAILABLE(macos(10.6), ios(7.0));

/*!
@function
@abstract Enables sampling profiler.
@param group The JavaScript context group to start sampling.
@result The value of the enablement, true if the sampling profiler gets enabled, otherwise false.
@discussion Remote inspection is true by default.
*/
JS_EXPORT bool JSContextGroupEnableSamplingProfiler(JSContextGroupRef group) JSC_API_AVAILABLE(macos(14.2), ios(17.2));

/*!
@function
@abstract Disables sampling profiler.
@param group The JavaScript context group to stop sampling.
*/
JS_EXPORT void JSContextGroupDisableSamplingProfiler(JSContextGroupRef group) JSC_API_AVAILABLE(macos(14.2), ios(17.2));

/*!
@function
@abstract Gets sampling profiler output in JSON form and clears the sampling profiler records.
@param group The JavaScript context group whose sampling profile output is taken.
@result The sampling profiler output in JSON form. NULL if sampling profiler is not enabled ever before.
@discussion Calling this function clears the sampling data accumulated so far.
*/
JS_EXPORT JSStringRef JSContextGroupTakeSamplesFromSamplingProfiler(JSContextGroupRef group) JSC_API_AVAILABLE(macos(14.2), ios(17.2));

/*!
@function
@abstract Gets a whether or not remote inspection is enabled on the context.
@param ctx The JSGlobalContext whose setting you want to get.
@result The value of the setting, true if remote inspection is enabled, otherwise false.
@discussion Remote inspection is true by default.
*/
JS_EXPORT bool JSGlobalContextGetRemoteInspectionEnabled(JSGlobalContextRef ctx) JSC_API_DEPRECATED_WITH_REPLACEMENT("JSGlobalContextIsInspectable", macos(10.10, 13.3), ios(8.0, 16.4));

/*!
@function
@abstract Sets the remote inspection setting for a context.
@param ctx The JSGlobalContext that you want to change.
@param enabled The new remote inspection enabled setting for the context.
*/
JS_EXPORT void JSGlobalContextSetRemoteInspectionEnabled(JSGlobalContextRef ctx, bool enabled) JSC_API_DEPRECATED_WITH_REPLACEMENT("JSGlobalContextSetInspectable", macos(10.10, 13.3), ios(8.0, 16.4));

/*!
@function
@abstract Gets the include native call stack when reporting exceptions setting for a context.
@param ctx The JSGlobalContext whose setting you want to get.
@result The value of the setting, true if remote inspection is enabled, otherwise false.
@discussion This setting is true by default.
*/
JS_EXPORT bool JSGlobalContextGetIncludesNativeCallStackWhenReportingExceptions(JSGlobalContextRef ctx) JSC_API_AVAILABLE(macos(10.10), ios(8.0));

/*!
@function
@abstract Sets the include native call stack when reporting exceptions setting for a context.
@param ctx The JSGlobalContext that you want to change.
@param includesNativeCallStack The new value of the setting for the context.
*/
JS_EXPORT void JSGlobalContextSetIncludesNativeCallStackWhenReportingExceptions(JSGlobalContextRef ctx, bool includesNativeCallStack) JSC_API_AVAILABLE(macos(10.10), ios(8.0));

/*!
@function
@abstract Sets the unhandled promise rejection callback for a context.
@discussion Similar to window.addEventListener('unhandledrejection'), but for contexts not associated with a web view.
@param ctx The JSGlobalContext to set the callback on.
@param function The callback function to set, which receives the promise and rejection reason as arguments.
@param exception A pointer to a JSValueRef in which to store an exception, if any. Pass NULL if you do not care to store an exception.
*/
JS_EXPORT void JSGlobalContextSetUnhandledRejectionCallback(JSGlobalContextRef ctx, JSObjectRef function, JSValueRef* exception) JSC_API_AVAILABLE(macos(10.15.4), ios(13.4));

/*!
@function
@abstract Sets whether a context allows use of eval (or the Function constructor).
@param ctx The JSGlobalContext that you want to change.
@param enabled The new eval enabled setting for the context.
@param message The error message to display when user attempts to call eval (or the Function constructor). Pass NULL when setting enabled to true.
*/
JS_EXPORT void JSGlobalContextSetEvalEnabled(JSGlobalContextRef ctx, bool enabled, JSStringRef message) JSC_API_AVAILABLE(macos(12.3), ios(15.4));

#ifdef __cplusplus
}
#endif

#endif /* JSContextRefPrivate_h */
