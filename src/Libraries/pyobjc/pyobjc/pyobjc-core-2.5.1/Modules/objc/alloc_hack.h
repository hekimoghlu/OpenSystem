/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#ifndef PyObjC_ALLOC_HACK_H
#define PyObjC_ALLOC_HACK_H
/*!
 * @header   alloc_hack.h
 * @abstract Special wrappers for the +alloc method
 * @discussion
 *      This module defines custom wrappers for the +alloc method. These
 *      are needed for some classes on MacOS X 10.2 because those classes
 *      cause crashes when alloc is called using NSInvocation.
 *
 *      The issue seems to be fixed in MacOS X 10.3. We keep using these
 *      wrapers just in case the problem returns.
 */

/*!
 * @function PyObjC_InstallAllocHack
 * @abstract Register the custom wrappers with the bridge
 * @result Returns 0 on success, -1 on error
 * @discussion
 *    This function installs the custom wrappers with the super-call.h module.
 */
int PyObjC_InstallAllocHack(void);

#endif /* PyObjC_ALLOC_HACK_H */
