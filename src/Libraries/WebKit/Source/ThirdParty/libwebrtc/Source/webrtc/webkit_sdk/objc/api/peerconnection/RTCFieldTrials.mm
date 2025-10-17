/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#import "RTCFieldTrials.h"

#include <memory>

#import "base/RTCLogging.h"

#include "system_wrappers/include/field_trial.h"

NSString * const kRTCFieldTrialAudioSendSideBweKey = @"WebRTC-Audio-SendSideBwe";
NSString * const kRTCFieldTrialAudioForceNoTWCCKey = @"WebRTC-Audio-ForceNoTWCC";
NSString * const kRTCFieldTrialAudioForceABWENoTWCCKey = @"WebRTC-Audio-ABWENoTWCC";
NSString * const kRTCFieldTrialSendSideBweWithOverheadKey = @"WebRTC-SendSideBwe-WithOverhead";
NSString * const kRTCFieldTrialFlexFec03AdvertisedKey = @"WebRTC-FlexFEC-03-Advertised";
NSString * const kRTCFieldTrialFlexFec03Key = @"WebRTC-FlexFEC-03";
NSString * const kRTCFieldTrialH264HighProfileKey = @"WebRTC-H264HighProfile";
NSString * const kRTCFieldTrialMinimizeResamplingOnMobileKey =
    @"WebRTC-Audio-MinimizeResamplingOnMobile";
NSString * const kRTCFieldTrialEnabledValue = @"Enabled";

static std::unique_ptr<char[]> gFieldTrialInitString;

void RTCInitFieldTrialDictionary(NSDictionary<NSString *, NSString *> *fieldTrials) {
  if (!fieldTrials) {
    RTCLogWarning(@"No fieldTrials provided.");
    return;
  }
  // Assemble the keys and values into the field trial string.
  // We don't perform any extra format checking. That should be done by the underlying WebRTC calls.
  NSMutableString *fieldTrialInitString = [NSMutableString string];
  for (NSString *key in fieldTrials) {
    NSString *fieldTrialEntry = [NSString stringWithFormat:@"%@/%@/", key, fieldTrials[key]];
    [fieldTrialInitString appendString:fieldTrialEntry];
  }
  size_t len = fieldTrialInitString.length + 1;
  gFieldTrialInitString.reset(new char[len]);
  if (![fieldTrialInitString getCString:gFieldTrialInitString.get()
                              maxLength:len
                               encoding:NSUTF8StringEncoding]) {
    RTCLogError(@"Failed to convert field trial string.");
    return;
  }
  webrtc::field_trial::InitFieldTrialsFromString(gFieldTrialInitString.get());
}
