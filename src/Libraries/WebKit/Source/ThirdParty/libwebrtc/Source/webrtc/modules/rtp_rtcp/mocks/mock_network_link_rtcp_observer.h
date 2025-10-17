/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#ifndef MODULES_RTP_RTCP_MOCKS_MOCK_NETWORK_LINK_RTCP_OBSERVER_H_
#define MODULES_RTP_RTCP_MOCKS_MOCK_NETWORK_LINK_RTCP_OBSERVER_H_

#include "api/array_view.h"
#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/include/report_block_data.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"
#include "test/gmock.h"

namespace webrtc {

class MockNetworkLinkRtcpObserver : public NetworkLinkRtcpObserver {
 public:
  MOCK_METHOD(void,
              OnRttUpdate,
              (Timestamp receive_time, TimeDelta rtt),
              (override));
  MOCK_METHOD(void,
              OnTransportFeedback,
              (Timestamp receive_time, const rtcp::TransportFeedback& feedback),
              (override));
  MOCK_METHOD(void,
              OnCongestionControlFeedback,
              (Timestamp receive_time,
               const rtcp::CongestionControlFeedback& feedback),
              (override));
  MOCK_METHOD(void,
              OnReceiverEstimatedMaxBitrate,
              (Timestamp receive_time, DataRate bitrate),
              (override));
  MOCK_METHOD(void,
              OnReport,
              (Timestamp receive_time,
               rtc::ArrayView<const ReportBlockData> report_blocks),
              (override));
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_MOCKS_MOCK_NETWORK_LINK_RTCP_OBSERVER_H_
