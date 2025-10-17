/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef JSStringRefBSTR_h
#define JSStringRefBSTR_h

#include "JSBase.h"

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/* COM convenience methods */

/*!
@function
@abstract         Creates a JavaScript string from a BSTR.
@param string     The BSTR to copy into the new JSString.
@result           A JSString containing string. Ownership follows the Create Rule.
*/
JS_EXPORT JSStringRef JSStringCreateWithBSTR(const BSTR string);

/*!
@function
@abstract         Creates a BSTR from a JavaScript string.
@param string     The JSString to copy into the new BSTR.
@result           A BSTR containing string. Ownership follows the Create Rule.
*/
JS_EXPORT BSTR JSStringCopyBSTR(const JSStringRef string);
    
#ifdef __cplusplus
}
#endif

#endif /* JSStringRefBSTR_h */
