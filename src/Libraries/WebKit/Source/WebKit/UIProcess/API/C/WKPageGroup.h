/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
#ifndef WKPageGroup_h
#define WKPageGroup_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKUserContentInjectedFrames.h>
#include <WebKit/WKUserScriptInjectionTime.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKPageGroupGetTypeID(void) WK_C_API_DEPRECATED;

WK_EXPORT WKPageGroupRef WKPageGroupCreateWithIdentifier(WKStringRef identifier) WK_C_API_DEPRECATED;

WK_EXPORT void WKPageGroupSetPreferences(WKPageGroupRef pageGroup, WKPreferencesRef preferences) WK_C_API_DEPRECATED;
WK_EXPORT WKPreferencesRef WKPageGroupGetPreferences(WKPageGroupRef pageGroup) WK_C_API_DEPRECATED;

WK_EXPORT WKUserContentControllerRef WKPageGroupGetUserContentController(WKPageGroupRef pageGroup) WK_C_API_DEPRECATED;

WK_EXPORT void WKPageGroupAddUserStyleSheet(WKPageGroupRef pageGroup, WKStringRef source, WKURLRef baseURL, WKArrayRef allowedURLPatterns, WKArrayRef blockedURLPatterns, WKUserContentInjectedFrames) WK_C_API_DEPRECATED;
WK_EXPORT void WKPageGroupRemoveAllUserStyleSheets(WKPageGroupRef pageGroup) WK_C_API_DEPRECATED;

WK_EXPORT void WKPageGroupAddUserScript(WKPageGroupRef pageGroup, WKStringRef source, WKURLRef baseURL, WKArrayRef allowedURLPatterns, WKArrayRef blockedURLPatterns, WKUserContentInjectedFrames, _WKUserScriptInjectionTime) WK_C_API_DEPRECATED;
WK_EXPORT void WKPageGroupRemoveAllUserScripts(WKPageGroupRef pageGroup) WK_C_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif /* WKPageGroup_h */
