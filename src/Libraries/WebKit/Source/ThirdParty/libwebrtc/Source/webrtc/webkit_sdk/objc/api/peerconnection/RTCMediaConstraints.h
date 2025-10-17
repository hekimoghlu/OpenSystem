/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

/** Constraint keys for media sources. */
/** The value for this key should be a base64 encoded string containing
 *  the data from the serialized configuration proto.
 */
RTC_EXTERN NSString *const kRTCMediaConstraintsAudioNetworkAdaptorConfig;

/** Constraint keys for generating offers and answers. */
RTC_EXTERN NSString *const kRTCMediaConstraintsIceRestart;
RTC_EXTERN NSString *const kRTCMediaConstraintsOfferToReceiveAudio;
RTC_EXTERN NSString *const kRTCMediaConstraintsOfferToReceiveVideo;
RTC_EXTERN NSString *const kRTCMediaConstraintsVoiceActivityDetection;

/** Constraint values for Boolean parameters. */
RTC_EXTERN NSString *const kRTCMediaConstraintsValueTrue;
RTC_EXTERN NSString *const kRTCMediaConstraintsValueFalse;

RTC_OBJC_EXPORT
@interface RTCMediaConstraints : NSObject

- (instancetype)init NS_UNAVAILABLE;

/** Initialize with mandatory and/or optional constraints. */
- (instancetype)
    initWithMandatoryConstraints:(nullable NSDictionary<NSString *, NSString *> *)mandatory
             optionalConstraints:(nullable NSDictionary<NSString *, NSString *> *)optional
    NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
