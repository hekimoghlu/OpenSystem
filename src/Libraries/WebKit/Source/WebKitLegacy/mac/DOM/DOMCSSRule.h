/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#import <WebKitLegacy/DOMObject.h>

@class DOMCSSRule;
@class DOMCSSStyleSheet;
@class NSString;

enum {
    DOM_UNKNOWN_RULE = 0,
    DOM_STYLE_RULE = 1,
    DOM_CHARSET_RULE = 2,
    DOM_IMPORT_RULE = 3,
    DOM_MEDIA_RULE = 4,
    DOM_FONT_FACE_RULE = 5,
    DOM_PAGE_RULE = 6,
    DOM_KEYFRAMES_RULE = 7,
    DOM_KEYFRAME_RULE = 8,
    DOM_NAMESPACE_RULE = 10,
    DOM_SUPPORTS_RULE = 12,
    DOM_WEBKIT_REGION_RULE = 16,
    DOM_WEBKIT_KEYFRAMES_RULE = 7,
    DOM_WEBKIT_KEYFRAME_RULE = 8
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMCSSRule : DOMObject
@property (readonly) unsigned short type;
@property (copy) NSString *cssText;
@property (readonly, strong) DOMCSSStyleSheet *parentStyleSheet;
@property (readonly, strong) DOMCSSRule *parentRule;
@end
