/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#ifndef WKNavigationDataRef_h
#define WKNavigationDataRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKNavigationDataGetTypeID(void);

WK_EXPORT WKStringRef WKNavigationDataCopyTitle(WKNavigationDataRef navigationData);
WK_EXPORT WKURLRequestRef WKNavigationDataCopyOriginalRequest(WKNavigationDataRef navigationData);
WK_EXPORT WKURLRef WKNavigationDataCopyNavigationDestinationURL(WKNavigationDataRef navigationDataRef);

// This returns the URL of the original request for backwards-compatibility purposes.
WK_EXPORT WKURLRef WKNavigationDataCopyURL(WKNavigationDataRef navigationData);

#ifdef __cplusplus
}
#endif

#endif /* WKNavigationDataRef_h */
