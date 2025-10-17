/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#import "RTCPeerConnection+Private.h"

#import "RTCLegacyStatsReport+Private.h"
#import "RTCMediaStreamTrack+Private.h"
#import "RTCRtpReceiver+Private.h"
#import "RTCRtpSender+Private.h"
#import "RTCStatisticsReport+Private.h"
#import "helpers/NSString+StdString.h"

#include "rtc_base/checks.h"

namespace webrtc {

class StatsCollectorCallbackAdapter : public RTCStatsCollectorCallback {
 public:
  StatsCollectorCallbackAdapter(RTCStatisticsCompletionHandler completion_handler)
      : completion_handler_(completion_handler) {}

  void OnStatsDelivered(const rtc::scoped_refptr<const RTCStatsReport> &report) override {
    RTC_DCHECK(completion_handler_);
    RTCStatisticsReport *statisticsReport = [[RTCStatisticsReport alloc] initWithReport:*report];
    completion_handler_(statisticsReport);
    completion_handler_ = nil;
  }

 private:
  RTCStatisticsCompletionHandler completion_handler_;
};

class StatsObserverAdapter : public StatsObserver {
 public:
  StatsObserverAdapter(void (^completionHandler)
      (NSArray<RTCLegacyStatsReport *> *stats)) {
    completion_handler_ = completionHandler;
  }

  ~StatsObserverAdapter() override { completion_handler_ = nil; }

  void OnComplete(const StatsReports& reports) override {
    RTC_DCHECK(completion_handler_);
    NSMutableArray *stats = [NSMutableArray arrayWithCapacity:reports.size()];
    for (const auto* report : reports) {
      RTCLegacyStatsReport *statsReport =
          [[RTCLegacyStatsReport alloc] initWithNativeReport:*report];
      [stats addObject:statsReport];
    }
    completion_handler_(stats);
    completion_handler_ = nil;
  }

 private:
  void (^completion_handler_)(NSArray<RTCLegacyStatsReport *> *stats);
};
}  // namespace webrtc

@implementation RTCPeerConnection (Stats)

- (void)statisticsForSender:(RTCRtpSender *)sender
          completionHandler:(RTCStatisticsCompletionHandler)completionHandler {
  rtc::scoped_refptr<webrtc::StatsCollectorCallbackAdapter> collector(
      new rtc::RefCountedObject<webrtc::StatsCollectorCallbackAdapter>(completionHandler));
  self.nativePeerConnection->GetStats(sender.nativeRtpSender, collector);
}

- (void)statisticsForReceiver:(RTCRtpReceiver *)receiver
            completionHandler:(RTCStatisticsCompletionHandler)completionHandler {
  rtc::scoped_refptr<webrtc::StatsCollectorCallbackAdapter> collector(
      new rtc::RefCountedObject<webrtc::StatsCollectorCallbackAdapter>(completionHandler));
  self.nativePeerConnection->GetStats(receiver.nativeRtpReceiver, collector);
}

- (void)statisticsWithCompletionHandler:(RTCStatisticsCompletionHandler)completionHandler {
  rtc::scoped_refptr<webrtc::StatsCollectorCallbackAdapter> collector(
      new rtc::RefCountedObject<webrtc::StatsCollectorCallbackAdapter>(completionHandler));
  self.nativePeerConnection->GetStats(collector);
}

- (void)statsForTrack:(RTCMediaStreamTrack *)mediaStreamTrack
     statsOutputLevel:(RTCStatsOutputLevel)statsOutputLevel
    completionHandler:
    (void (^)(NSArray<RTCLegacyStatsReport *> *stats))completionHandler {
  rtc::scoped_refptr<webrtc::StatsObserverAdapter> observer(
      new rtc::RefCountedObject<webrtc::StatsObserverAdapter>
          (completionHandler));
  webrtc::PeerConnectionInterface::StatsOutputLevel nativeOutputLevel =
      [[self class] nativeStatsOutputLevelForLevel:statsOutputLevel];
  self.nativePeerConnection->GetStats(
      observer, mediaStreamTrack.nativeTrack, nativeOutputLevel);
}

@end
