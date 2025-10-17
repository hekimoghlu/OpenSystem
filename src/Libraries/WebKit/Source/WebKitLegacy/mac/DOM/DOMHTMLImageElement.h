/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
@interface DOMHTMLImageElement : DOMHTMLElement
@property (copy) NSString *name;
@property (copy) NSString *align;
@property (copy) NSString *alt;
@property (copy) NSString *border;
@property int height;
@property int hspace;
@property BOOL isMap;
@property (copy) NSString *longDesc;
@property (copy) NSString *src;
@property (copy) NSString *useMap;
@property int vspace;
@property int width;
@property (readonly) BOOL complete WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *lowsrc WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int naturalHeight WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int naturalWidth WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int x WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int y WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *altDisplayString WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSURL *absoluteImageURL WEBKIT_AVAILABLE_MAC(10_5);
@end
