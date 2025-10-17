/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#ifndef API_DTLS_TRANSPORT_INTERFACE_H_
#define API_DTLS_TRANSPORT_INTERFACE_H_

#include <memory>
#include <optional>

#include "absl/base/attributes.h"
#include "api/ice_transport_interface.h"
#include "api/ref_count.h"
#include "api/rtc_error.h"
#include "api/scoped_refptr.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// States of a DTLS transport, corresponding to the JS API specification.
// http://w3c.github.io/webrtc-pc/#dom-rtcdtlstransportstate
enum class DtlsTransportState {
  kNew,         // Has not started negotiating yet.
  kConnecting,  // In the process of negotiating a secure connection.
  kConnected,   // Completed negotiation and verified fingerprints.
  kClosed,      // Intentionally closed.
  kFailed,      // Failure due to an error or failing to verify a remote
                // fingerprint.
  kNumValues
};

enum class DtlsTransportTlsRole {
  kServer,  // Other end sends CLIENT_HELLO
  kClient   // This end sends CLIENT_HELLO
};

// This object gives snapshot information about the changeable state of a
// DTLSTransport.
class RTC_EXPORT DtlsTransportInformation {
 public:
  DtlsTransportInformation();
  explicit DtlsTransportInformation(DtlsTransportState state);
  DtlsTransportInformation(
      DtlsTransportState state,
      std::optional<DtlsTransportTlsRole> role,
      std::optional<int> tls_version,
      std::optional<int> ssl_cipher_suite,
      std::optional<int> srtp_cipher_suite,
      std::unique_ptr<rtc::SSLCertChain> remote_ssl_certificates);
  ABSL_DEPRECATED("Use version with role parameter")
  DtlsTransportInformation(
      DtlsTransportState state,
      std::optional<int> tls_version,
      std::optional<int> ssl_cipher_suite,
      std::optional<int> srtp_cipher_suite,
      std::unique_ptr<rtc::SSLCertChain> remote_ssl_certificates);

  // Copy and assign
  DtlsTransportInformation(const DtlsTransportInformation& c);
  DtlsTransportInformation& operator=(const DtlsTransportInformation& c);
  // Move
  DtlsTransportInformation(DtlsTransportInformation&& other) = default;
  DtlsTransportInformation& operator=(DtlsTransportInformation&& other) =
      default;

  DtlsTransportState state() const { return state_; }
  std::optional<DtlsTransportTlsRole> role() const { return role_; }
  std::optional<int> tls_version() const { return tls_version_; }
  std::optional<int> ssl_cipher_suite() const { return ssl_cipher_suite_; }
  std::optional<int> srtp_cipher_suite() const { return srtp_cipher_suite_; }
  // The accessor returns a temporary pointer, it does not release ownership.
  const rtc::SSLCertChain* remote_ssl_certificates() const {
    return remote_ssl_certificates_.get();
  }

 private:
  DtlsTransportState state_;
  std::optional<DtlsTransportTlsRole> role_;
  std::optional<int> tls_version_;
  std::optional<int> ssl_cipher_suite_;
  std::optional<int> srtp_cipher_suite_;
  std::unique_ptr<rtc::SSLCertChain> remote_ssl_certificates_;
};

class DtlsTransportObserverInterface {
 public:
  // This callback carries information about the state of the transport.
  // The argument is a pass-by-value snapshot of the state.
  virtual void OnStateChange(DtlsTransportInformation info) = 0;
  // This callback is called when an error occurs, causing the transport
  // to go to the kFailed state.
  virtual void OnError(RTCError error) = 0;

 protected:
  virtual ~DtlsTransportObserverInterface() = default;
};

// A DTLS transport, as represented to the outside world.
// This object is created on the network thread, and can only be
// accessed on that thread, except for functions explicitly marked otherwise.
// References can be held by other threads, and destruction can therefore
// be initiated by other threads.
class DtlsTransportInterface : public webrtc::RefCountInterface {
 public:
  // Returns a pointer to the ICE transport that is owned by the DTLS transport.
  virtual rtc::scoped_refptr<IceTransportInterface> ice_transport() = 0;
  // Returns information on the state of the DtlsTransport.
  // This function can be called from other threads.
  virtual DtlsTransportInformation Information() = 0;
  // Observer management.
  virtual void RegisterObserver(DtlsTransportObserverInterface* observer) = 0;
  virtual void UnregisterObserver() = 0;
};

}  // namespace webrtc

#endif  // API_DTLS_TRANSPORT_INTERFACE_H_
