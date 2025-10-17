/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#ifndef PC_PEER_CONNECTION_MESSAGE_HANDLER_H_
#define PC_PEER_CONNECTION_MESSAGE_HANDLER_H_

#include <functional>

#include "api/jsep.h"
#include "api/legacy_stats_types.h"
#include "api/media_stream_interface.h"
#include "api/peer_connection_interface.h"
#include "api/rtc_error.h"
#include "api/task_queue/pending_task_safety_flag.h"
#include "api/task_queue/task_queue_base.h"
#include "pc/legacy_stats_collector_interface.h"

namespace webrtc {

class PeerConnectionMessageHandler {
 public:
  explicit PeerConnectionMessageHandler(rtc::Thread* signaling_thread)
      : signaling_thread_(signaling_thread) {}
  ~PeerConnectionMessageHandler() = default;

  void PostSetSessionDescriptionSuccess(
      SetSessionDescriptionObserver* observer);
  void PostSetSessionDescriptionFailure(SetSessionDescriptionObserver* observer,
                                        RTCError&& error);
  void PostCreateSessionDescriptionFailure(
      CreateSessionDescriptionObserver* observer,
      RTCError error);
  void PostGetStats(StatsObserver* observer,
                    LegacyStatsCollectorInterface* legacy_stats,
                    MediaStreamTrackInterface* track);
  void RequestUsagePatternReport(std::function<void()>, int delay_ms);

 private:
  ScopedTaskSafety safety_;
  TaskQueueBase* const signaling_thread_;
};

}  // namespace webrtc

#endif  // PC_PEER_CONNECTION_MESSAGE_HANDLER_H_
