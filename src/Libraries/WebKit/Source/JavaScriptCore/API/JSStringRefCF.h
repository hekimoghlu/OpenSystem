/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#ifndef JSStringRefCF_h
#define JSStringRefCF_h

#include <CoreFoundation/CoreFoundation.h>
#include <JavaScriptCore/JSBase.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CFString convenience methods */

/*!
@function
@abstract         Creates a JavaScript string from a CFString.
@discussion       This function is optimized to take advantage of cases when 
 CFStringGetCharactersPtr returns a valid pointer.
@param string     The CFString to copy into the new JSString.
@result           A JSString containing string. Ownership follows the Create Rule.
*/
JS_EXPORT JSStringRef JSStringCreateWithCFString(CFStringRef string);
/*!
@function
@abstract         Creates a CFString from a JavaScript string.
@param alloc      The alloc parameter to pass to CFStringCreate.
@param string     The JSString to copy into the new CFString.
@result           A CFString containing string. Ownership follows the Create Rule.
*/
JS_EXPORT CFStringRef JSStringCopyCFString(CFAllocatorRef alloc, JSStringRef string) CF_RETURNS_RETAINED;

#ifdef __cplusplus
}
#endif

#endif /* JSStringRefCF_h */
