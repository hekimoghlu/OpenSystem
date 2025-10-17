/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "test/pc/e2e/stats_poller.h"

#include <utility>

#include "rtc_base/logging.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
namespace webrtc_pc_e2e {

void InternalStatsObserver::PollStats() {
  peer_->GetStats(this);
}

void InternalStatsObserver::OnStatsDelivered(
    const rtc::scoped_refptr<const RTCStatsReport>& report) {
  for (auto* observer : observers_) {
    observer->OnStatsReports(pc_label_, report);
  }
}

StatsPoller::StatsPoller(std::vector<StatsObserverInterface*> observers,
                         std::map<std::string, StatsProvider*> peers)
    : observers_(std::move(observers)) {
  webrtc::MutexLock lock(&mutex_);
  for (auto& peer : peers) {
    pollers_.push_back(rtc::make_ref_counted<InternalStatsObserver>(
        peer.first, peer.second, observers_));
  }
}

StatsPoller::StatsPoller(std::vector<StatsObserverInterface*> observers,
                         std::map<std::string, TestPeer*> peers)
    : observers_(std::move(observers)) {
  webrtc::MutexLock lock(&mutex_);
  for (auto& peer : peers) {
    pollers_.push_back(rtc::make_ref_counted<InternalStatsObserver>(
        peer.first, peer.second, observers_));
  }
}

void StatsPoller::PollStatsAndNotifyObservers() {
  webrtc::MutexLock lock(&mutex_);
  for (auto& poller : pollers_) {
    poller->PollStats();
  }
}

void StatsPoller::RegisterParticipantInCall(absl::string_view peer_name,
                                            StatsProvider* peer) {
  webrtc::MutexLock lock(&mutex_);
  pollers_.push_back(rtc::make_ref_counted<InternalStatsObserver>(
      peer_name, peer, observers_));
}

bool StatsPoller::UnregisterParticipantInCall(absl::string_view peer_name) {
  webrtc::MutexLock lock(&mutex_);
  for (auto it = pollers_.begin(); it != pollers_.end(); ++it) {
    if ((*it)->pc_label() == peer_name) {
      pollers_.erase(it);
      return true;
    }
  }
  return false;
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
