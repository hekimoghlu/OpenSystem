/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#ifndef JSContextRefInternal_h
#define JSContextRefInternal_h

#include "JSContextRefPrivate.h"

#if USE(CF)
#include <CoreFoundation/CFRunLoop.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if USE(CF)
/*!
@function
@abstract Gets the run loop used by the Web Inspector debugger when evaluating JavaScript in this context.
@param ctx The JSGlobalContext whose setting you want to get.
*/
JS_EXPORT CFRunLoopRef JSGlobalContextGetDebuggerRunLoop(JSGlobalContextRef ctx) JSC_API_AVAILABLE(macos(10.10), ios(8.0));

/*!
@function
@abstract Sets the run loop used by the Web Inspector debugger when evaluating JavaScript in this context.
@param ctx The JSGlobalContext that you want to change.
@param runLoop The new value of the setting for the context.
*/
JS_EXPORT void JSGlobalContextSetDebuggerRunLoop(JSGlobalContextRef ctx, CFRunLoopRef runLoop) JSC_API_AVAILABLE(macos(10.10), ios(8.0));
#endif

#ifdef __cplusplus
}
#endif

#endif // JSContextRefInternal_h
