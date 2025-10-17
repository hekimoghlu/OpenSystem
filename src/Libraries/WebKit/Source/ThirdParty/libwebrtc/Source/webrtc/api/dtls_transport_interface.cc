/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#include "api/dtls_transport_interface.h"

#include <memory>
#include <optional>
#include <utility>

#include "rtc_base/ssl_certificate.h"

namespace webrtc {

DtlsTransportInformation::DtlsTransportInformation()
    : state_(DtlsTransportState::kNew) {}

DtlsTransportInformation::DtlsTransportInformation(DtlsTransportState state)
    : state_(state) {}

DtlsTransportInformation::DtlsTransportInformation(
    DtlsTransportState state,
    std::optional<DtlsTransportTlsRole> role,
    std::optional<int> tls_version,
    std::optional<int> ssl_cipher_suite,
    std::optional<int> srtp_cipher_suite,
    std::unique_ptr<rtc::SSLCertChain> remote_ssl_certificates)
    : state_(state),
      role_(role),
      tls_version_(tls_version),
      ssl_cipher_suite_(ssl_cipher_suite),
      srtp_cipher_suite_(srtp_cipher_suite),
      remote_ssl_certificates_(std::move(remote_ssl_certificates)) {}

// Deprecated version
DtlsTransportInformation::DtlsTransportInformation(
    DtlsTransportState state,
    std::optional<int> tls_version,
    std::optional<int> ssl_cipher_suite,
    std::optional<int> srtp_cipher_suite,
    std::unique_ptr<rtc::SSLCertChain> remote_ssl_certificates)
    : state_(state),
      role_(std::nullopt),
      tls_version_(tls_version),
      ssl_cipher_suite_(ssl_cipher_suite),
      srtp_cipher_suite_(srtp_cipher_suite),
      remote_ssl_certificates_(std::move(remote_ssl_certificates)) {}

DtlsTransportInformation::DtlsTransportInformation(
    const DtlsTransportInformation& c)
    : state_(c.state()),
      role_(c.role_),
      tls_version_(c.tls_version_),
      ssl_cipher_suite_(c.ssl_cipher_suite_),
      srtp_cipher_suite_(c.srtp_cipher_suite_),
      remote_ssl_certificates_(c.remote_ssl_certificates()
                                   ? c.remote_ssl_certificates()->Clone()
                                   : nullptr) {}

DtlsTransportInformation& DtlsTransportInformation::operator=(
    const DtlsTransportInformation& c) {
  state_ = c.state();
  role_ = c.role_;
  tls_version_ = c.tls_version_;
  ssl_cipher_suite_ = c.ssl_cipher_suite_;
  srtp_cipher_suite_ = c.srtp_cipher_suite_;
  remote_ssl_certificates_ = c.remote_ssl_certificates()
                                 ? c.remote_ssl_certificates()->Clone()
                                 : nullptr;
  return *this;
}

}  // namespace webrtc
