/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#ifndef WKSerializedScriptValue_h
#define WKSerializedScriptValue_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKSerializedScriptValueGetTypeID(void) WK_C_API_DEPRECATED;

WK_EXPORT WKSerializedScriptValueRef WKSerializedScriptValueCreate(JSContextRef context, JSValueRef value, JSValueRef* exception) WK_C_API_DEPRECATED;
WK_EXPORT JSValueRef WKSerializedScriptValueDeserialize(WKSerializedScriptValueRef scriptValue, JSContextRef context, JSValueRef* exception) WK_C_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif /* WKSerializedScriptValue_h */
