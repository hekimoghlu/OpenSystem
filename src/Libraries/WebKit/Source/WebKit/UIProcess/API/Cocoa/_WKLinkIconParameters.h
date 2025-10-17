/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

typedef NS_ENUM(NSInteger, WKLinkIconType) {
    WKLinkIconTypeFavicon,
    WKLinkIconTypeTouchIcon,
    WKLinkIconTypeTouchPrecomposedIcon,
} WK_API_AVAILABLE(macos(10.12.4), ios(10.3));

WK_CLASS_AVAILABLE(macos(10.12.4), ios(10.3))
@interface _WKLinkIconParameters : NSObject

@property (nonatomic, readonly, copy) NSURL *url;
@property (nonatomic, readonly) WKLinkIconType iconType;
@property (nonatomic, readonly, copy) NSString *mimeType;
@property (nonatomic, readonly, copy) NSNumber *size;

@property (nonatomic, readonly, copy) NSDictionary *attributes WK_API_AVAILABLE(macos(10.14), ios(12.0));

@end
