/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#ifndef PC_DTLS_SRTP_TRANSPORT_H_
#define PC_DTLS_SRTP_TRANSPORT_H_

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "api/dtls_transport_interface.h"
#include "api/rtc_error.h"
#include "p2p/base/dtls_transport_internal.h"
#include "p2p/base/packet_transport_internal.h"
#include "pc/srtp_transport.h"
#include "rtc_base/buffer.h"

namespace webrtc {

// The subclass of SrtpTransport is used for DTLS-SRTP. When the DTLS handshake
// is finished, it extracts the keying materials from DtlsTransport and
// configures the SrtpSessions in the base class.
class DtlsSrtpTransport : public SrtpTransport {
 public:
  DtlsSrtpTransport(bool rtcp_mux_enabled, const FieldTrialsView& field_trials);

  // Set P2P layer RTP/RTCP DtlsTransports. When using RTCP-muxing,
  // `rtcp_dtls_transport` is null.
  void SetDtlsTransports(cricket::DtlsTransportInternal* rtp_dtls_transport,
                         cricket::DtlsTransportInternal* rtcp_dtls_transport);

  void SetRtcpMuxEnabled(bool enable) override;

  // Set the header extension ids that should be encrypted.
  void UpdateSendEncryptedHeaderExtensionIds(
      const std::vector<int>& send_extension_ids);

  void UpdateRecvEncryptedHeaderExtensionIds(
      const std::vector<int>& recv_extension_ids);

  void SetOnDtlsStateChange(std::function<void(void)> callback);

  // If `active_reset_srtp_params_` is set to be true, the SRTP parameters will
  // be reset whenever the DtlsTransports are reset.
  void SetActiveResetSrtpParams(bool active_reset_srtp_params) {
    active_reset_srtp_params_ = active_reset_srtp_params;
  }

 private:
  bool IsDtlsActive();
  bool IsDtlsConnected();
  bool IsDtlsWritable();
  bool DtlsHandshakeCompleted();
  void MaybeSetupDtlsSrtp();
  void SetupRtpDtlsSrtp();
  void SetupRtcpDtlsSrtp();
  bool ExtractParams(cricket::DtlsTransportInternal* dtls_transport,
                     int* selected_crypto_suite,
                     rtc::ZeroOnFreeBuffer<uint8_t>* send_key,
                     rtc::ZeroOnFreeBuffer<uint8_t>* recv_key);
  void SetDtlsTransport(cricket::DtlsTransportInternal* new_dtls_transport,
                        cricket::DtlsTransportInternal** old_dtls_transport);
  void SetRtpDtlsTransport(cricket::DtlsTransportInternal* rtp_dtls_transport);
  void SetRtcpDtlsTransport(
      cricket::DtlsTransportInternal* rtcp_dtls_transport);

  void OnDtlsState(cricket::DtlsTransportInternal* dtls_transport,
                   DtlsTransportState state);

  // Override the SrtpTransport::OnWritableState.
  void OnWritableState(rtc::PacketTransportInternal* packet_transport) override;

  // Owned by the TransportController.
  cricket::DtlsTransportInternal* rtp_dtls_transport_ = nullptr;
  cricket::DtlsTransportInternal* rtcp_dtls_transport_ = nullptr;

  // The encrypted header extension IDs.
  std::optional<std::vector<int>> send_extension_ids_;
  std::optional<std::vector<int>> recv_extension_ids_;

  bool active_reset_srtp_params_ = false;
  std::function<void(void)> on_dtls_state_change_;
};

}  // namespace webrtc

#endif  // PC_DTLS_SRTP_TRANSPORT_H_
