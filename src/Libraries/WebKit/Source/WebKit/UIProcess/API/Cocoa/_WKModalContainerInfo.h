/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#import <WebKit/WKFoundation.h>

typedef NS_OPTIONS(NSUInteger, _WKModalContainerControlTypes) {
    _WKModalContainerControlTypeNeutral = 1 << 0,
    _WKModalContainerControlTypePositive = 1 << 1,
    _WKModalContainerControlTypeNegative = 1 << 2,
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

WK_CLASS_AVAILABLE(macos(13.0), ios(16.0))
@interface _WKModalContainerInfo : NSObject

@property (nonatomic, readonly) _WKModalContainerControlTypes availableTypes;

@end
