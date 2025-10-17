/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#ifndef P2P_BASE_TRANSPORT_DESCRIPTION_FACTORY_H_
#define P2P_BASE_TRANSPORT_DESCRIPTION_FACTORY_H_

#include <memory>
#include <utility>

#include "api/field_trials_view.h"
#include "p2p/base/ice_credentials_iterator.h"
#include "p2p/base/transport_description.h"
#include "rtc_base/rtc_certificate.h"

namespace rtc {
class SSLIdentity;
}

namespace cricket {

struct TransportOptions {
  bool ice_restart = false;
  bool prefer_passive_role = false;
  // If true, ICE renomination is supported and will be used if it is also
  // supported by the remote side.
  bool enable_ice_renomination = false;
};

// Creates transport descriptions according to the supplied configuration.
// When creating answers, performs the appropriate negotiation
// of the various fields to determine the proper result.
class TransportDescriptionFactory {
 public:
  // Default ctor; use methods below to set configuration.
  explicit TransportDescriptionFactory(
      const webrtc::FieldTrialsView& field_trials);
  ~TransportDescriptionFactory();

  // The certificate to use when setting up DTLS.
  const rtc::scoped_refptr<rtc::RTCCertificate>& certificate() const {
    return certificate_;
  }

  // Specifies the certificate to use
  void set_certificate(rtc::scoped_refptr<rtc::RTCCertificate> certificate) {
    certificate_ = std::move(certificate);
  }

  // Creates a transport description suitable for use in an offer.
  std::unique_ptr<TransportDescription> CreateOffer(
      const TransportOptions& options,
      const TransportDescription* current_description,
      IceCredentialsIterator* ice_credentials) const;
  // Create a transport description that is a response to an offer.
  //
  // If `require_transport_attributes` is true, then TRANSPORT category
  // attributes are expected to be present in `offer`, as defined by
  // sdp-mux-attributes, and null will be returned otherwise. It's expected
  // that this will be set to false for an m= section that's in a BUNDLE group
  // but isn't the first m= section in the group.
  std::unique_ptr<TransportDescription> CreateAnswer(
      const TransportDescription* offer,
      const TransportOptions& options,
      bool require_transport_attributes,
      const TransportDescription* current_description,
      IceCredentialsIterator* ice_credentials) const;

  const webrtc::FieldTrialsView& trials() const { return field_trials_; }
  // Functions for disabling encryption - test only!
  // In insecure mode, the connection will accept a description without
  // fingerprint, and will generate SDP even if certificate is not set.
  // If certificate is set, it will accept a description both with and
  // without fingerprint, but will generate a description with fingerprint.
  bool insecure() const { return insecure_; }
  void SetInsecureForTesting() { insecure_ = true; }

 private:
  bool SetSecurityInfo(TransportDescription* description,
                       ConnectionRole role) const;
  bool insecure_ = false;
  rtc::scoped_refptr<rtc::RTCCertificate> certificate_;
  const webrtc::FieldTrialsView& field_trials_;
};

}  // namespace cricket

#endif  // P2P_BASE_TRANSPORT_DESCRIPTION_FACTORY_H_
