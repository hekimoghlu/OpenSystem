/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebKitAvailability.h>

@class DOMEvent;
@class NSString;
@protocol DOMEventListener;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@protocol DOMEventTarget <NSObject, NSCopying>
- (void)addEventListener:(NSString *)type listener:(id <DOMEventListener>)listener useCapture:(BOOL)useCapture WEBKIT_AVAILABLE_MAC(10_5);
- (void)removeEventListener:(NSString *)type listener:(id <DOMEventListener>)listener useCapture:(BOOL)useCapture WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)dispatchEvent:(DOMEvent *)event;
- (void)addEventListener:(NSString *)type :(id <DOMEventListener>)listener :(BOOL)useCapture WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)removeEventListener:(NSString *)type :(id <DOMEventListener>)listener :(BOOL)useCapture WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
