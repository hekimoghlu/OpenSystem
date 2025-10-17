/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#ifndef WKViewPrivate_h
#define WKViewPrivate_h

#if TARGET_OS_IPHONE

#import "WKView.h"

#ifdef __cplusplus
extern "C" {
#endif    

void _WKViewSetWindow (WKViewRef view, WAKWindow *window);
void _WKViewSetSuperview (WKViewRef view, WKViewRef superview);
void _WKViewWillRemoveSubview(WKViewRef view, WKViewRef subview);
WKViewRef _WKViewBaseView (WKViewRef view);
void _WKViewAutoresize(WKViewRef view, const CGRect *oldSuperFrame, const CGRect *newSuperFrame);
void _WKViewSetViewContext (WKViewRef view, WKViewContext *context);

#ifdef __cplusplus
}
#endif

#endif // TARGET_OS_IPHONE

#endif // WKViewPrivate_h
