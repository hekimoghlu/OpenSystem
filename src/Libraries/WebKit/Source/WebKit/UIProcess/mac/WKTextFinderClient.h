/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#ifndef WKTextFinderClient_h
#define WKTextFinderClient_h

#import <WebKit/WKFoundation.h>
#import <wtf/NakedRef.h>

#if PLATFORM(MAC)

OBJC_CLASS NSView;

namespace WebKit {
class WebPageProxy;
}

@interface WKTextFinderClient : NSObject

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view usePlatformFindUI:(BOOL)usePlatformFindUI;
- (void)willDestroyView:(NSView *)view;

@end

#endif // PLATFORM(MAC)

#endif // WKTextFinderClient_h
