/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

@class DOMAbstractView;
@class DOMDocument;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMHTMLIFrameElement : DOMHTMLElement
@property (copy) NSString *align;
@property (copy) NSString *frameBorder;
@property (copy) NSString *height;
@property (copy) NSString *longDesc;
@property (copy) NSString *marginHeight;
@property (copy) NSString *marginWidth;
@property (copy) NSString *name;
@property (copy) NSString *scrolling;
@property (copy) NSString *src;
@property (copy) NSString *width;
@property (readonly, strong) DOMDocument *contentDocument;
@property (readonly, strong) DOMAbstractView *contentWindow WEBKIT_AVAILABLE_MAC(10_6);
@end
