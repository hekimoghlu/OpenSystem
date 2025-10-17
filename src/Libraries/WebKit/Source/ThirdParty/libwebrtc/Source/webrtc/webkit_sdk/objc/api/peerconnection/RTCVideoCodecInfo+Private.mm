/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#import "RTCVideoCodecInfo+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTCVideoCodecInfo (Private)

- (instancetype)initWithNativeSdpVideoFormat:(webrtc::SdpVideoFormat)format {
  NSMutableDictionary *params = [NSMutableDictionary dictionary];
  for (auto it = format.parameters.begin(); it != format.parameters.end(); ++it) {
    [params setObject:[NSString rtcStringForStdString:it->second]
               forKey:[NSString rtcStringForStdString:it->first]];
  }
  return [self initWithName:[NSString rtcStringForStdString:format.name] parameters:params];
}

- (webrtc::SdpVideoFormat)nativeSdpVideoFormat {
  std::map<std::string, std::string> parameters;
  for (NSString *paramKey in self.parameters.allKeys) {
    std::string key = [NSString rtcStdStringForString:paramKey];
    std::string value = [NSString rtcStdStringForString:self.parameters[paramKey]];
    parameters[key] = value;
  }

  return webrtc::SdpVideoFormat([NSString rtcStdStringForString:self.name], parameters);
}

@end
