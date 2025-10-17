/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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

#include "api/stats/rtc_stats_collector_callback.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

using ::testing::Eq;

class TestStatsProvider : public StatsProvider {
 public:
  ~TestStatsProvider() override = default;

  void GetStats(RTCStatsCollectorCallback* callback) override {
    stats_collections_count_++;
  }

  int stats_collections_count() const { return stats_collections_count_; }

 private:
  int stats_collections_count_ = 0;
};

class MockStatsObserver : public StatsObserverInterface {
 public:
  ~MockStatsObserver() override = default;

  MOCK_METHOD(void,
              OnStatsReports,
              (absl::string_view pc_label,
               const rtc::scoped_refptr<const RTCStatsReport>& report));
};

TEST(StatsPollerTest, UnregisterParticipantAddedInCtor) {
  TestStatsProvider alice;
  TestStatsProvider bob;

  MockStatsObserver stats_observer;

  StatsPoller poller(/*observers=*/{&stats_observer},
                     /*peers_to_observe=*/{{"alice", &alice}, {"bob", &bob}});
  poller.PollStatsAndNotifyObservers();

  EXPECT_THAT(alice.stats_collections_count(), Eq(1));
  EXPECT_THAT(bob.stats_collections_count(), Eq(1));

  poller.UnregisterParticipantInCall("bob");
  poller.PollStatsAndNotifyObservers();

  EXPECT_THAT(alice.stats_collections_count(), Eq(2));
  EXPECT_THAT(bob.stats_collections_count(), Eq(1));
}

TEST(StatsPollerTest, UnregisterParticipantRegisteredInCall) {
  TestStatsProvider alice;
  TestStatsProvider bob;

  MockStatsObserver stats_observer;

  StatsPoller poller(/*observers=*/{&stats_observer},
                     /*peers_to_observe=*/{{"alice", &alice}});
  poller.RegisterParticipantInCall("bob", &bob);
  poller.PollStatsAndNotifyObservers();

  EXPECT_THAT(alice.stats_collections_count(), Eq(1));
  EXPECT_THAT(bob.stats_collections_count(), Eq(1));

  poller.UnregisterParticipantInCall("bob");
  poller.PollStatsAndNotifyObservers();

  EXPECT_THAT(alice.stats_collections_count(), Eq(2));
  EXPECT_THAT(bob.stats_collections_count(), Eq(1));
}

}  // namespace
}  // namespace webrtc_pc_e2e
}  // namespace webrtc
