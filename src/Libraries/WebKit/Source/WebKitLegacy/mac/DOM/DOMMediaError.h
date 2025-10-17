/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

enum {
    DOM_MEDIA_ERR_ABORTED = 1,
    DOM_MEDIA_ERR_NETWORK = 2,
    DOM_MEDIA_ERR_DECODE = 3,
    DOM_MEDIA_ERR_SRC_NOT_SUPPORTED = 4,
    DOM_MEDIA_ERR_ENCRYPTED = 5
} WEBKIT_ENUM_DEPRECATED_MAC(10_5, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_5, 10_14)
@interface DOMMediaError : DOMObject
@property (readonly) unsigned short code;
@end
