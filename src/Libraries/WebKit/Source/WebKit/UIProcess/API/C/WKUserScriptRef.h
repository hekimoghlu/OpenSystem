/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#ifndef WKUserScriptRef_h
#define WKUserScriptRef_h

#include <WebKit/WKBase.h>
#include <WebKit/WKUserScriptInjectionTime.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKUserScriptGetTypeID(void);

WK_EXPORT WKUserScriptRef WKUserScriptCreate(WKStringRef source, WKURLRef url, WKArrayRef includeURLPatterns, WKArrayRef excludeURLPatterns, _WKUserScriptInjectionTime injectionTime, bool forMainFrameOnly);
WK_EXPORT WKUserScriptRef WKUserScriptCreateWithSource(WKStringRef source, _WKUserScriptInjectionTime injectionTime, bool forMainFrameOnly);

WK_EXPORT WKStringRef WKUserScriptCopySource(WKUserScriptRef userScript);
WK_EXPORT _WKUserScriptInjectionTime WKUserScriptGetInjectionTime(WKUserScriptRef userScript);
WK_EXPORT bool WKUserScriptGetMainFrameOnly(WKUserScriptRef userScript);

#ifdef __cplusplus
}
#endif

#endif /* WKUserScriptRef_h */
