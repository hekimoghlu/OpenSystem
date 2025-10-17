/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

#if !TARGET_OS_IPHONE

NS_ASSUME_NONNULL_BEGIN

@class _WKHitTestResult;

WK_CLASS_AVAILABLE(macos(10.12))
@interface _WKContextMenuElementInfo : NSObject <NSCopying>

@property (nonatomic, readonly, copy) _WKHitTestResult *hitTestResult WK_API_AVAILABLE(macos(13.3));
@property (nonatomic, readonly, copy, nullable) NSString *qrCodePayloadString WK_API_AVAILABLE(macos(14.0));
@property (nonatomic, readonly) BOOL hasEntireImage WK_API_AVAILABLE(macos(14.0));

@end

NS_ASSUME_NONNULL_END

#endif
