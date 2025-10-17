/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
#ifndef WKView_h
#define WKView_h

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wpe_view_backend;
typedef struct _WPEDisplay WPEDisplay;
typedef struct _WPEView WPEView;

#if ENABLE(WPE_PLATFORM)
WK_EXPORT WKViewRef WKViewCreate(WPEDisplay*, WKPageConfigurationRef);
#endif
WK_EXPORT WKViewRef WKViewCreateDeprecated(struct wpe_view_backend*, WKPageConfigurationRef);

WK_EXPORT WKPageRef WKViewGetPage(WKViewRef);
#if ENABLE(WPE_PLATFORM)
WK_EXPORT WPEView*  WKViewGetView(WKViewRef);
#endif

#ifdef __cplusplus
}
#endif

#endif // WKView_h
