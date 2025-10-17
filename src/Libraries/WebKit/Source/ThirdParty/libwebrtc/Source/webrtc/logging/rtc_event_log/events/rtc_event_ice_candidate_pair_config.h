/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_ICE_CANDIDATE_PAIR_CONFIG_H_
#define LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_ICE_CANDIDATE_PAIR_CONFIG_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/candidate.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/units/timestamp.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {

enum class IceCandidatePairConfigType {
  kAdded,
  kUpdated,
  kDestroyed,
  kSelected,
  kNumValues,
};

enum class IceCandidatePairProtocol {
  kUnknown,
  kUdp,
  kTcp,
  kSsltcp,
  kTls,
  kNumValues,
};

enum class IceCandidatePairAddressFamily {
  kUnknown,
  kIpv4,
  kIpv6,
  kNumValues,
};

enum class IceCandidateNetworkType {
  kUnknown,
  kEthernet,
  kLoopback,
  kWifi,
  kVpn,
  kCellular,
  kNumValues,
};

struct LoggedIceCandidatePairConfig {
  int64_t log_time_us() const { return timestamp.us(); }
  int64_t log_time_ms() const { return timestamp.ms(); }
  Timestamp log_time() const { return timestamp; }

  Timestamp timestamp = Timestamp::MinusInfinity();
  IceCandidatePairConfigType type;
  uint32_t candidate_pair_id;
  IceCandidateType local_candidate_type;
  IceCandidatePairProtocol local_relay_protocol;
  IceCandidateNetworkType local_network_type;
  IceCandidatePairAddressFamily local_address_family;
  IceCandidateType remote_candidate_type;
  IceCandidatePairAddressFamily remote_address_family;
  IceCandidatePairProtocol candidate_pair_protocol;
};

class IceCandidatePairDescription {
 public:
  IceCandidatePairDescription(IceCandidateType local_candidate_type,
                              IceCandidateType remote_candidate_type);
  explicit IceCandidatePairDescription(
      const IceCandidatePairDescription& other);

  ~IceCandidatePairDescription();

  IceCandidateType local_candidate_type;
  IceCandidatePairProtocol local_relay_protocol;
  IceCandidateNetworkType local_network_type;
  IceCandidatePairAddressFamily local_address_family;
  IceCandidateType remote_candidate_type;
  IceCandidatePairAddressFamily remote_address_family;
  IceCandidatePairProtocol candidate_pair_protocol;
};

class RtcEventIceCandidatePairConfig final : public RtcEvent {
 public:
  static constexpr Type kType = Type::IceCandidatePairConfig;

  RtcEventIceCandidatePairConfig(
      IceCandidatePairConfigType type,
      uint32_t candidate_pair_id,
      const IceCandidatePairDescription& candidate_pair_desc);

  ~RtcEventIceCandidatePairConfig() override;

  Type GetType() const override { return kType; }
  // N.B. An ICE config event is not considered an RtcEventLog config event.
  bool IsConfigEvent() const override { return false; }

  std::unique_ptr<RtcEventIceCandidatePairConfig> Copy() const;

  IceCandidatePairConfigType type() const { return type_; }
  uint32_t candidate_pair_id() const { return candidate_pair_id_; }
  const IceCandidatePairDescription& candidate_pair_desc() const {
    return candidate_pair_desc_;
  }

  static std::string Encode(rtc::ArrayView<const RtcEvent*> /* batch */) {
    // TODO(terelius): Implement
    return "";
  }

  static RtcEventLogParseStatus Parse(
      absl::string_view /* encoded_bytes */,
      bool /* batched */,
      std::vector<LoggedIceCandidatePairConfig>& /* output */) {
    // TODO(terelius): Implement
    return RtcEventLogParseStatus::Error("Not Implemented", __FILE__, __LINE__);
  }

 private:
  RtcEventIceCandidatePairConfig(const RtcEventIceCandidatePairConfig& other);

  const IceCandidatePairConfigType type_;
  const uint32_t candidate_pair_id_;
  const IceCandidatePairDescription candidate_pair_desc_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_EVENTS_RTC_EVENT_ICE_CANDIDATE_PAIR_CONFIG_H_
