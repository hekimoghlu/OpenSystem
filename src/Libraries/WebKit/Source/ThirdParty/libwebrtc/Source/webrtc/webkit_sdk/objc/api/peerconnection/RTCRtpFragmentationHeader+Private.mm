/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#import "RTCRtpFragmentationHeader+Private.h"

#include "modules/include/module_common_types.h"

@implementation RTCRtpFragmentationHeader (Private)

- (instancetype)initWithNativeFragmentationHeader:
        (const webrtc::RTPFragmentationHeader *)fragmentationHeader {
  if (self = [super init]) {
    if (fragmentationHeader) {
      int count = fragmentationHeader->fragmentationVectorSize;
      NSMutableArray *offsets = [NSMutableArray array];
      NSMutableArray *lengths = [NSMutableArray array];
      NSMutableArray *timeDiffs = [NSMutableArray array];
      NSMutableArray *plTypes = [NSMutableArray array];
      for (int i = 0; i < count; ++i) {
        [offsets addObject:@(fragmentationHeader->fragmentationOffset[i])];
        [lengths addObject:@(fragmentationHeader->fragmentationLength[i])];
        [timeDiffs addObject:@(0)];
        [plTypes addObject:@(0)];
      }
      self.fragmentationOffset = [offsets copy];
      self.fragmentationLength = [lengths copy];
      self.fragmentationTimeDiff = [timeDiffs copy];
      self.fragmentationPlType = [plTypes copy];
    }
  }

  return self;
}

- (std::unique_ptr<webrtc::RTPFragmentationHeader>)createNativeFragmentationHeader {
  auto fragmentationHeader =
      std::unique_ptr<webrtc::RTPFragmentationHeader>(new webrtc::RTPFragmentationHeader);
  fragmentationHeader->VerifyAndAllocateFragmentationHeader(self.fragmentationOffset.count);
  for (NSUInteger i = 0; i < self.fragmentationOffset.count; ++i) {
    fragmentationHeader->fragmentationOffset[i] = (size_t)self.fragmentationOffset[i].unsignedIntValue;
    fragmentationHeader->fragmentationLength[i] = (size_t)self.fragmentationLength[i].unsignedIntValue;
  }

  return fragmentationHeader;
}

@end
