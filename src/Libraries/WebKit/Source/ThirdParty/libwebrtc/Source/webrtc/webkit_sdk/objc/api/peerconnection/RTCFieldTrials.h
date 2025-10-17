/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

/** The only valid value for the following if set is kRTCFieldTrialEnabledValue. */
RTC_EXTERN NSString * const kRTCFieldTrialAudioSendSideBweKey;
RTC_EXTERN NSString * const kRTCFieldTrialAudioForceNoTWCCKey;
RTC_EXTERN NSString * const kRTCFieldTrialAudioForceABWENoTWCCKey;
RTC_EXTERN NSString * const kRTCFieldTrialSendSideBweWithOverheadKey;
RTC_EXTERN NSString * const kRTCFieldTrialFlexFec03AdvertisedKey;
RTC_EXTERN NSString * const kRTCFieldTrialFlexFec03Key;
RTC_EXTERN NSString * const kRTCFieldTrialH264HighProfileKey;
RTC_EXTERN NSString * const kRTCFieldTrialMinimizeResamplingOnMobileKey;

/** The valid value for field trials above. */
RTC_EXTERN NSString * const kRTCFieldTrialEnabledValue;

/** Initialize field trials using a dictionary mapping field trial keys to their
 * values. See above for valid keys and values. Must be called before any other
 * call into WebRTC. See: webrtc/system_wrappers/include/field_trial.h
 */
RTC_EXTERN void RTCInitFieldTrialDictionary(NSDictionary<NSString *, NSString *> *fieldTrials);
