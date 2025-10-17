/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#ifndef JSContextPrivate_h
#define JSContextPrivate_h

#if JSC_OBJC_API_ENABLED

#import <JavaScriptCore/JSContext.h>

@protocol JSModuleLoaderDelegate <NSObject>

@required

/*! @abstract Provides source code for any JS module that is actively imported.
 @param context The context for which the module is being requested.
 @param identifier The resolved identifier for the requested module.
 @param resolve A JS function to call with the desired script for identifier.
 @param reject A JS function to call when identifier cannot be fetched.
 @discussion Currently, identifier will always be an absolute file URL computed from specifier of the requested module relative to the URL of the requesting script. If the requesting script does not have a URL and the module specifier is not an absolute path the module loader will fail to load the module.

 The first argument to resolve sholud always be a JSScript, otherwise the module loader will reject the module.

 Once an identifier has been resolved or rejected in a given context it will never be requested again. If a script is successfully evaluated it will not be re-evaluated on any subsequent import.

 The VM will retain all evaluated modules for the lifetime of the context.
 */
- (void)context:(JSContext *)context fetchModuleForIdentifier:(JSValue *)identifier withResolveHandler:(JSValue *)resolve andRejectHandler:(JSValue *)reject;

@optional

/*! @abstract This is called before the module with "key" is evaluated.
 @param key The module key for the module that is about to be evaluated.
 */
- (void)willEvaluateModule:(NSURL *)key;

/*! @abstract This is called after the module with "key" is evaluated.
 @param key The module key for the module that was just evaluated.
 */
- (void)didEvaluateModule:(NSURL *)key;

@end

@interface JSContext(Private)

/*!
@property
@discussion Remote inspection setting of the JSContext. Default value is YES.
*/
@property (setter=_setRemoteInspectionEnabled:) BOOL _remoteInspectionEnabled JSC_API_DEPRECATED_WITH_REPLACEMENT("inspectable", macos(10.10, 13.3), ios(8.0, 16.4));

/*!
@property
@discussion Set whether or not the native call stack is included when reporting exceptions. Default value is YES.
*/
@property (setter=_setIncludesNativeCallStackWhenReportingExceptions:) BOOL _includesNativeCallStackWhenReportingExceptions JSC_API_AVAILABLE(macos(10.10), ios(8.0));

/*!
@property
@discussion Set the run loop the Web Inspector debugger should use when evaluating JavaScript in the JSContext.
*/
@property (setter=_setDebuggerRunLoop:) CFRunLoopRef _debuggerRunLoop JSC_API_AVAILABLE(macos(10.10), ios(8.0));

/*! @abstract The delegate the context will use when trying to load a module. Note, this delegate will be ignored for contexts returned by UIWebView. */
@property (nonatomic, weak) id <JSModuleLoaderDelegate> moduleLoaderDelegate JSC_API_AVAILABLE(macos(10.15), ios(13.0));

/*!
 @method
 @abstract Run a JSScript.
 @param script the JSScript to evaluate.
 @discussion If the provided JSScript was created with kJSScriptTypeProgram, the script will run synchronously and return the result of evaluation.

 Otherwise, if the script was created with kJSScriptTypeModule, the module will be run asynchronously and will return a promise resolved when the module and any transitive dependencies are loaded. The module loader will treat the script as if it had been returned from a delegate call to moduleLoaderDelegate. This mirrors the JavaScript dynamic import operation.
 */
// FIXME: Before making this public need to fix: https://bugs.webkit.org/show_bug.cgi?id=199714
- (JSValue *)evaluateJSScript:(JSScript *)script JSC_API_AVAILABLE(macos(10.15), ios(13.0));

/*!
 @method
 @abstract Get the identifiers of the modules a JSScript depends on in this context.
 @param script the JSScript to determine the dependencies of.
 @result An Array containing all the identifiers of modules script depends on.
 @discussion If the provided JSScript was not created with kJSScriptTypeModule, an exception will be thrown. Also, if the script has not already been evaluated in this context an error will be throw.
 */
- (JSValue *)dependencyIdentifiersForModuleJSScript:(JSScript *)script JSC_API_AVAILABLE(macos(10.15), ios(13.0));

/*!
 @method
 @abstract Mark this JSContext as an ITMLKit context for the purposes of remote inspection capabilities.
 */
- (void)_setITMLDebuggableType JSC_API_AVAILABLE(macos(11.0), ios(14.0));

@end

#endif

#endif // JSContextInternal_h
