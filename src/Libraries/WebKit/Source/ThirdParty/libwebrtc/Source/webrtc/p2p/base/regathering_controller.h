/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
#ifndef P2P_BASE_REGATHERING_CONTROLLER_H_
#define P2P_BASE_REGATHERING_CONTROLLER_H_

#include <memory>

#include "api/task_queue/pending_task_safety_flag.h"
#include "p2p/base/ice_transport_internal.h"
#include "p2p/base/port_allocator.h"
#include "rtc_base/thread.h"

namespace webrtc {

// Controls regathering of candidates for the ICE transport passed into it,
// reacting to signals like SignalWritableState, SignalNetworkRouteChange, etc.,
// using methods like GetStats to get additional information, and calling
// methods like RegatherOnFailedNetworks on the PortAllocatorSession when
// regathering is desired.
//
// "Regathering" is defined as gathering additional candidates within a single
// ICE generation (or in other words, PortAllocatorSession), and is possible
// when "continual gathering" is enabled. This may allow connectivity to be
// maintained and/or restored without a full ICE restart.
//
// Regathering will only begin after PortAllocationSession is set via
// set_allocator_session. This should be called any time the "active"
// PortAllocatorSession is changed (in other words, when an ICE restart occurs),
// so that candidates are gathered for the "current" ICE generation.
//
// All methods of BasicRegatheringController should be called on the same
// thread as the one passed to the constructor, and this thread should be the
// same one where PortAllocatorSession runs, which is also identical to the
// network thread of the ICE transport, as given by
// P2PTransportChannel::thread().
class BasicRegatheringController : public sigslot::has_slots<> {
 public:
  struct Config {
    int regather_on_failed_networks_interval =
        cricket::REGATHER_ON_FAILED_NETWORKS_INTERVAL;
  };

  BasicRegatheringController() = delete;
  BasicRegatheringController(const Config& config,
                             cricket::IceTransportInternal* ice_transport,
                             rtc::Thread* thread);
  ~BasicRegatheringController() override;
  // TODO(qingsi): Remove this method after implementing a new signal in
  // P2PTransportChannel and reacting to that signal for the initial schedules
  // of regathering.
  void Start();
  void set_allocator_session(cricket::PortAllocatorSession* allocator_session) {
    allocator_session_ = allocator_session;
  }
  // Setting a different config of the regathering interval range on all
  // networks cancels and reschedules the recurring schedules, if any, of
  // regathering on all networks. The same applies to the change of the
  // regathering interval on the failed networks. This rescheduling behavior is
  // seperately defined for the two config parameters.
  void SetConfig(const Config& config);

 private:
  // TODO(qingsi): Implement the following methods and use methods from the ICE
  // transport like GetStats to get additional information for the decision
  // making in regathering.
  void OnIceTransportStateChanged(cricket::IceTransportInternal*) {}
  void OnIceTransportWritableState(rtc::PacketTransportInternal*) {}
  void OnIceTransportReceivingState(rtc::PacketTransportInternal*) {}
  void OnIceTransportNetworkRouteChanged(std::optional<rtc::NetworkRoute>) {}
  // Schedules delayed and repeated regathering of local candidates on failed
  // networks, where the delay in milliseconds is given by the config. Each
  // repetition is separated by the same delay. When scheduled, all previous
  // schedules are canceled.
  void ScheduleRecurringRegatheringOnFailedNetworks();
  // Cancels regathering scheduled by ScheduleRecurringRegatheringOnAllNetworks.
  void CancelScheduledRecurringRegatheringOnAllNetworks();

  // We use a flag to be able to cancel pending regathering operations when
  // the object goes out of scope or the config changes.
  std::unique_ptr<ScopedTaskSafety> pending_regathering_;
  Config config_;
  cricket::IceTransportInternal* ice_transport_;
  cricket::PortAllocatorSession* allocator_session_ = nullptr;
  rtc::Thread* const thread_;
};

}  // namespace webrtc

#endif  // P2P_BASE_REGATHERING_CONTROLLER_H_
