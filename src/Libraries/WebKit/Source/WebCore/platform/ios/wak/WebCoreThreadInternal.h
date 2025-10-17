/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#ifndef WebCoreThreadInternal_h
#define WebCoreThreadInternal_h

#include "WebCoreThread.h"

#if defined(__cplusplus)
extern "C" {
#endif    

// Sometimes, like for the Inspector, we need to pause the execution of a current run
// loop iteration and resume it later. This handles pushing and popping the autorelease
// pools to keep the original pool unaffected by the run loop observers. The
// WebThreadLock is released when calling Enable, and acquired when calling Disable.
// NOTE: Does not expect arbitrary nesting, only 1 level of nesting.
void WebRunLoopEnableNested();
void WebRunLoopDisableNested();

void WebThreadInitRunQueue();

WEBCORE_EXPORT CFRunLoopRef WebThreadRunLoop(void);
WebThreadContext *WebThreadCurrentContext(void);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // WebCoreThreadInternal_h
