/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#ifndef PC_TRANSPORT_STATS_H_
#define PC_TRANSPORT_STATS_H_

#include <string>
#include <vector>

#include "api/dtls_transport_interface.h"
#include "p2p/base/dtls_transport_internal.h"
#include "p2p/base/ice_transport_internal.h"
#include "p2p/base/port.h"
#include "rtc_base/ssl_stream_adapter.h"

namespace cricket {

struct TransportChannelStats {
  TransportChannelStats();
  TransportChannelStats(const TransportChannelStats&);
  ~TransportChannelStats();

  int component = 0;
  int ssl_version_bytes = 0;
  int srtp_crypto_suite = rtc::kSrtpInvalidCryptoSuite;
  int ssl_cipher_suite = rtc::kTlsNullWithNullNull;
  std::optional<absl::string_view> tls_cipher_suite_name;
  std::optional<rtc::SSLRole> dtls_role;
  webrtc::DtlsTransportState dtls_state = webrtc::DtlsTransportState::kNew;
  IceTransportStats ice_transport_stats;
  uint16_t ssl_peer_signature_algorithm = rtc::kSslSignatureAlgorithmUnknown;
};

// Information about all the channels of a transport.
// TODO(hta): Consider if a simple vector is as good as a map.
typedef std::vector<TransportChannelStats> TransportChannelStatsList;

// Information about the stats of a transport.
struct TransportStats {
  std::string transport_name;
  TransportChannelStatsList channel_stats;
};

}  // namespace cricket

#endif  // PC_TRANSPORT_STATS_H_
