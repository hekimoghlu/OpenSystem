/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#import <WebKitLegacy/DOMHTMLElement.h>

@class NSString;
@class NSURL;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLAreaElement : DOMHTMLElement
@property (copy) NSString *alt;
@property (copy) NSString *coords;
@property BOOL noHref;
@property (copy) NSString *shape;
@property (copy) NSString *target;
@property (copy) NSString *accessKey WEBKIT_DEPRECATED_MAC(10_4, 10_8);
@property (readonly, copy) NSURL *absoluteLinkURL WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *href;
@property (readonly, copy) NSString *protocol WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *host WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *hostname WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *port WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *pathname WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *search WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *hashName WEBKIT_AVAILABLE_MAC(10_5);
@end
