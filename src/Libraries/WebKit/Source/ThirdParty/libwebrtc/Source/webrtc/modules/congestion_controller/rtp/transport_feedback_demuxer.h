/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#ifndef MODULES_CONGESTION_CONTROLLER_RTP_TRANSPORT_FEEDBACK_DEMUXER_H_
#define MODULES_CONGESTION_CONTROLLER_RTP_TRANSPORT_FEEDBACK_DEMUXER_H_

#include <map>
#include <utility>
#include <vector>

#include "api/sequence_checker.h"
#include "modules/include/module_common_types_public.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"
#include "rtc_base/system/no_unique_address.h"

namespace webrtc {

// Implementation of StreamFeedbackProvider that provides a way for
// implementations of StreamFeedbackObserver to register for feedback callbacks
// for a given set of SSRCs.
// Registration methods need to be called from the same execution context
// (thread or task queue) and callbacks to
// StreamFeedbackObserver::OnPacketFeedbackVector will be made in that same
// context.
// TODO(tommi): This appears to be the only implementation of this interface.
// Do we need the interface?
class TransportFeedbackDemuxer final : public StreamFeedbackProvider {
 public:
  TransportFeedbackDemuxer();

  // Implements StreamFeedbackProvider interface
  void RegisterStreamFeedbackObserver(
      std::vector<uint32_t> ssrcs,
      StreamFeedbackObserver* observer) override;
  void DeRegisterStreamFeedbackObserver(
      StreamFeedbackObserver* observer) override;
  void AddPacket(const RtpPacketSendInfo& packet_info);
  void OnTransportFeedback(const rtcp::TransportFeedback& feedback);

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker observer_checker_;
  RtpSequenceNumberUnwrapper seq_num_unwrapper_
      RTC_GUARDED_BY(&observer_checker_);
  std::map<int64_t, StreamFeedbackObserver::StreamPacketInfo> history_
      RTC_GUARDED_BY(&observer_checker_);

  // Maps a set of ssrcs to corresponding observer. Vectors are used rather than
  // set/map to ensure that the processing order is consistent independently of
  // the randomized ssrcs.
  std::vector<std::pair<std::vector<uint32_t>, StreamFeedbackObserver*>>
      observers_ RTC_GUARDED_BY(&observer_checker_);
};
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_RTP_TRANSPORT_FEEDBACK_DEMUXER_H_
