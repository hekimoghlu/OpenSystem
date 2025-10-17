/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "p2p/base/regathering_controller.h"

#include "api/task_queue/pending_task_safety_flag.h"
#include "api/units/time_delta.h"

namespace webrtc {

BasicRegatheringController::BasicRegatheringController(
    const Config& config,
    cricket::IceTransportInternal* ice_transport,
    rtc::Thread* thread)
    : config_(config), ice_transport_(ice_transport), thread_(thread) {
  RTC_DCHECK(thread_);
  RTC_DCHECK_RUN_ON(thread_);
  RTC_DCHECK(ice_transport_);
  ice_transport_->SignalStateChanged.connect(
      this, &BasicRegatheringController::OnIceTransportStateChanged);
  ice_transport->SignalWritableState.connect(
      this, &BasicRegatheringController::OnIceTransportWritableState);
  ice_transport->SignalReceivingState.connect(
      this, &BasicRegatheringController::OnIceTransportReceivingState);
  ice_transport->SignalNetworkRouteChanged.connect(
      this, &BasicRegatheringController::OnIceTransportNetworkRouteChanged);
}

BasicRegatheringController::~BasicRegatheringController() {
  RTC_DCHECK_RUN_ON(thread_);
}

void BasicRegatheringController::Start() {
  RTC_DCHECK_RUN_ON(thread_);
  ScheduleRecurringRegatheringOnFailedNetworks();
}

void BasicRegatheringController::SetConfig(const Config& config) {
  RTC_DCHECK_RUN_ON(thread_);
  bool need_reschedule_on_failed_networks =
      pending_regathering_ && (config_.regather_on_failed_networks_interval !=
                               config.regather_on_failed_networks_interval);
  config_ = config;
  if (need_reschedule_on_failed_networks) {
    ScheduleRecurringRegatheringOnFailedNetworks();
  }
}

void BasicRegatheringController::
    ScheduleRecurringRegatheringOnFailedNetworks() {
  RTC_DCHECK_RUN_ON(thread_);
  RTC_DCHECK(config_.regather_on_failed_networks_interval >= 0);
  // Reset pending_regathering_ to cancel any potentially pending tasks.
  pending_regathering_.reset(new ScopedTaskSafety());

  thread_->PostDelayedTask(
      SafeTask(pending_regathering_->flag(),
               [this]() {
                 RTC_DCHECK_RUN_ON(thread_);
                 // Only regather when the current session is in the CLEARED
                 // state (i.e., not running or stopped). It is only
                 // possible to enter this state when we gather continually,
                 // so there is an implicit check on continual gathering
                 // here.
                 if (allocator_session_ && allocator_session_->IsCleared()) {
                   allocator_session_->RegatherOnFailedNetworks();
                 }
                 ScheduleRecurringRegatheringOnFailedNetworks();
               }),
      TimeDelta::Millis(config_.regather_on_failed_networks_interval));
}

}  // namespace webrtc
