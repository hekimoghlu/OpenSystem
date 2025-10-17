/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#ifndef WKUserContentURLPattern_h
#define WKUserContentURLPattern_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKUserContentURLPatternGetTypeID(void);

WK_EXPORT WKUserContentURLPatternRef WKUserContentURLPatternCreate(WKStringRef patternRef);

WK_EXPORT WKStringRef WKUserContentURLPatternCopyHost(WKUserContentURLPatternRef urlPatternRef);
WK_EXPORT WKStringRef WKUserContentURLPatternCopyScheme(WKUserContentURLPatternRef urlPatternRef);
WK_EXPORT bool WKUserContentURLPatternIsValid(WKUserContentURLPatternRef urlPatternRef);
WK_EXPORT bool WKUserContentURLPatternMatchesURL(WKUserContentURLPatternRef urlPatternRef, WKURLRef urlRef);
WK_EXPORT bool WKUserContentURLPatternMatchesSubdomains(WKUserContentURLPatternRef urlPatternRef);

#ifdef __cplusplus
}
#endif

#endif /* WKUserContentURLPattern_h */
