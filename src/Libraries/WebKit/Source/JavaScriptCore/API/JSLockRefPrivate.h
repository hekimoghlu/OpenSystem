/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include <JavaScriptCore/JSBase.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 @function
 @abstract Acquire the API lock for the given JSContextRef.
 @param ctx The execution context to be locked.
 @discussion The lock has to be held to perform any interactions with the JSContextRef. This function allows holding the lock across multiple interactions to amortize the cost. This lock is a recursive lock.
 */
JS_EXPORT void JSLock(JSContextRef ctx);

/*!
 @function
 @abstract Release the API lock for the given JSContextRef.
 @param ctx The execution context to be unlocked.
 @discussion Releases the lock that was previously acquired using JSLock.
 */
JS_EXPORT void JSUnlock(JSContextRef ctx);

#ifdef __cplusplus
}
#endif
