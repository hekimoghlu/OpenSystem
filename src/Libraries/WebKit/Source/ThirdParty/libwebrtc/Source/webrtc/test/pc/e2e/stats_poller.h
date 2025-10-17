/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#ifndef TEST_PC_E2E_STATS_POLLER_H_
#define TEST_PC_E2E_STATS_POLLER_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "api/peer_connection_interface.h"
#include "api/stats/rtc_stats_collector_callback.h"
#include "api/test/stats_observer_interface.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "test/pc/e2e/stats_provider.h"
#include "test/pc/e2e/test_peer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// Helper class that will notify all the webrtc::test::StatsObserverInterface
// objects subscribed.
class InternalStatsObserver : public RTCStatsCollectorCallback {
 public:
  InternalStatsObserver(absl::string_view pc_label,
                        StatsProvider* peer,
                        std::vector<StatsObserverInterface*> observers)
      : pc_label_(pc_label), peer_(peer), observers_(std::move(observers)) {}

  std::string pc_label() const { return pc_label_; }

  void PollStats();

  void OnStatsDelivered(
      const rtc::scoped_refptr<const RTCStatsReport>& report) override;

 private:
  std::string pc_label_;
  StatsProvider* peer_;
  std::vector<StatsObserverInterface*> observers_;
};

// Helper class to invoke GetStats on a PeerConnection by passing a
// webrtc::StatsObserver that will notify all the
// webrtc::test::StatsObserverInterface subscribed.
class StatsPoller {
 public:
  StatsPoller(std::vector<StatsObserverInterface*> observers,
              std::map<std::string, StatsProvider*> peers_to_observe);
  StatsPoller(std::vector<StatsObserverInterface*> observers,
              std::map<std::string, TestPeer*> peers_to_observe);

  void PollStatsAndNotifyObservers();

  void RegisterParticipantInCall(absl::string_view peer_name,
                                 StatsProvider* peer);
  // Unregister participant from stats poller. Returns true if participant was
  // removed and false if participant wasn't found.
  bool UnregisterParticipantInCall(absl::string_view peer_name);

 private:
  const std::vector<StatsObserverInterface*> observers_;
  webrtc::Mutex mutex_;
  std::vector<rtc::scoped_refptr<InternalStatsObserver>> pollers_
      RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_STATS_POLLER_H_
