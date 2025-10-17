/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTC_OBJC_TYPE (RTCRtpHeaderExtensionCapability) : NSObject

/** The URI of the RTP header extension, as defined in RFC5285. */
@property(nonatomic, readonly, copy) NSString *uri;

/** The value put in the RTP packet to identify the header extension. */
@property(nonatomic, readonly, nullable) NSNumber* preferredId;

/** Whether the header extension is encrypted or not. */
@property(nonatomic, readonly, getter=isPreferredEncrypted) BOOL preferredEncrypted;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
