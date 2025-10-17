/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#import "RTCH264ProfileLevelId.h"

#import "helpers/NSString+StdString.h"
#if defined(WEBRTC_IOS)
#import "UIDevice+H264Profile.h"
#endif

#include "api/video_codecs/h264_profile_level_id.h"
#include "media/base/media_constants.h"

namespace {

#if !defined(WEBRTC_WEBKIT_BUILD)
NSString *MaxSupportedProfileLevelConstrainedHigh();
NSString *MaxSupportedProfileLevelConstrainedBaseline();
#endif

}  // namespace

NSString *const kRTCVideoCodecH264Name = @"H264";
NSString *const kRTCLevel31ConstrainedHigh = @"640c1f";
NSString *const kRTCLevel31ConstrainedBaseline = @"42e01f";

#if defined(WEBRTC_WEBKIT_BUILD)
NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedHigh =
    @"640c1f";
NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedBaseline =
    @"42e01f";
#else
NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedHigh =
    MaxSupportedProfileLevelConstrainedHigh();
NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedBaseline =
    MaxSupportedProfileLevelConstrainedBaseline();
#endif

namespace {

#if !defined(WEBRTC_WEBKIT_BUILD)

#if defined(WEBRTC_IOS)

using namespace webrtc::H264;

NSString *MaxSupportedLevelForProfile(Profile profile) {
  const std::optional<ProfileLevelId> profileLevelId = [UIDevice maxSupportedH264Profile];
  if (profileLevelId && profileLevelId->profile >= profile) {
    const std::optional<std::string> profileString =
        ProfileLevelIdToString(ProfileLevelId(profile, profileLevelId->level));
    if (profileString) {
      return [NSString rtcStringForStdString:*profileString];
    }
  }
  return nil;
}
#endif

NSString *MaxSupportedProfileLevelConstrainedBaseline() {
#if defined(WEBRTC_IOS)
  NSString *profile = MaxSupportedLevelForProfile(webrtc::H264::kProfileConstrainedBaseline);
  if (profile != nil) {
    return profile;
  }
#endif
  return kRTCLevel31ConstrainedBaseline;
}

NSString *MaxSupportedProfileLevelConstrainedHigh() {
#if defined(WEBRTC_IOS)
  NSString *profile = MaxSupportedLevelForProfile(webrtc::H264::kProfileConstrainedHigh);
  if (profile != nil) {
    return profile;
  }
#endif
  return kRTCLevel31ConstrainedHigh;
}

#endif

}  // namespace

@interface RTCH264ProfileLevelId ()

@property(nonatomic, assign) RTCH264Profile profile;
@property(nonatomic, assign) RTCH264Level level;
@property(nonatomic, strong) NSString *hexString;

@end

@implementation RTCH264ProfileLevelId

@synthesize profile = _profile;
@synthesize level = _level;
@synthesize hexString = _hexString;

- (instancetype)initWithHexString:(NSString *)hexString {
  if (self = [super init]) {
    self.hexString = hexString;

      std::optional<webrtc::H264ProfileLevelId> profile_level_id =
        webrtc::ParseH264ProfileLevelId([hexString cStringUsingEncoding:NSUTF8StringEncoding]);
    if (profile_level_id.has_value()) {
      self.profile = static_cast<RTCH264Profile>(profile_level_id->profile);
      self.level = static_cast<RTCH264Level>(profile_level_id->level);
    }
  }
  return self;
}

- (instancetype)initWithProfile:(RTCH264Profile)profile level:(RTCH264Level)level {
  if (self = [super init]) {
    self.profile = profile;
    self.level = level;

      std::optional<std::string> hex_string =
        webrtc::H264ProfileLevelIdToString(webrtc::H264ProfileLevelId(
            static_cast<webrtc::H264Profile>(profile), static_cast<webrtc::H264Level>(level)));
    self.hexString =
        [NSString stringWithCString:hex_string.value_or("").c_str() encoding:NSUTF8StringEncoding];
  }
  return self;
}

@end
