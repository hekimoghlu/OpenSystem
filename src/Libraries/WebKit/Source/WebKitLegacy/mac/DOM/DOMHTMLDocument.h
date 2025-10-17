/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#import <WebKitLegacy/DOMDocument.h>

@class DOMHTMLCollection;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLDocument : DOMDocument
@property (readonly, strong) DOMHTMLCollection *embeds WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMHTMLCollection *plugins WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMHTMLCollection *scripts WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int width WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int height WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *dir WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *designMode WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *compatMode WEBKIT_AVAILABLE_MAC(10_6);
@property (copy) NSString *bgColor WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *fgColor WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *alinkColor WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *linkColor WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *vlinkColor WEBKIT_AVAILABLE_MAC(10_5);

- (void)open;
- (void)close;
- (void)write:(NSString *)text;
- (void)writeln:(NSString *)text;
- (void)clear WEBKIT_AVAILABLE_MAC(10_6);
- (void)captureEvents WEBKIT_AVAILABLE_MAC(10_5);
- (void)releaseEvents WEBKIT_AVAILABLE_MAC(10_5);
@end
