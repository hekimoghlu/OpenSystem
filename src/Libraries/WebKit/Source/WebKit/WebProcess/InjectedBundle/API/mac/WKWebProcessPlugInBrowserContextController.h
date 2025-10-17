/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

@class WKDOMDocument;
@class WKDOMRange;
@class WKWebProcessPlugInFrame;
@protocol WKWebProcessPlugInLoadDelegate;

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKWebProcessPlugInBrowserContextController : NSObject

@property (readonly) WKDOMDocument *mainFrameDocument;

@property (readonly) WKDOMRange *selectedRange;

@property (readonly) WKWebProcessPlugInFrame *mainFrame WK_API_DEPRECATED("With site isolation, the main frame is not necessarily local to the current bundle process.", macos(10.10, 15.2), ios(8.0, 18.2));

@property (weak) id <WKWebProcessPlugInLoadDelegate> loadDelegate;

@end
