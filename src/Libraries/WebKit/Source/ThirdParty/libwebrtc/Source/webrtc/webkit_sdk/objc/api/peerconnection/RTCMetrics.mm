/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#import "RTCMetrics.h"

#import "RTCMetricsSampleInfo+Private.h"

void RTCEnableMetrics(void) {
  webrtc::metrics::Enable();
}

NSArray<RTCMetricsSampleInfo *> *RTCGetAndResetMetrics(void) {
  std::map<std::string, std::unique_ptr<webrtc::metrics::SampleInfo>>
      histograms;
  webrtc::metrics::GetAndReset(&histograms);

  NSMutableArray *metrics =
      [NSMutableArray arrayWithCapacity:histograms.size()];
  for (auto const &histogram : histograms) {
    RTCMetricsSampleInfo *metric = [[RTCMetricsSampleInfo alloc]
        initWithNativeSampleInfo:*histogram.second];
    [metrics addObject:metric];
  }
  return metrics;
}
