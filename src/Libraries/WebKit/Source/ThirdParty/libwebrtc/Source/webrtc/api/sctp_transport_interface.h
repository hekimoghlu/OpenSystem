/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef API_SCTP_TRANSPORT_INTERFACE_H_
#define API_SCTP_TRANSPORT_INTERFACE_H_

#include <optional>

#include "api/dtls_transport_interface.h"
#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// States of a SCTP transport, corresponding to the JS API specification.
// http://w3c.github.io/webrtc-pc/#dom-rtcsctptransportstate
enum class SctpTransportState {
  kNew,         // Has not started negotiating yet. Non-standard state.
  kConnecting,  // In the process of negotiating an association.
  kConnected,   // Completed negotiation of an association.
  kClosed,      // Closed by local or remote party.
  kNumValues
};

// This object gives snapshot information about the changeable state of a
// SctpTransport.
// It reflects the readonly attributes of the object in the specification.
// http://w3c.github.io/webrtc-pc/#rtcsctptransport-interface
class RTC_EXPORT SctpTransportInformation {
 public:
  SctpTransportInformation() = default;
  SctpTransportInformation(const SctpTransportInformation&) = default;
  explicit SctpTransportInformation(SctpTransportState state);
  SctpTransportInformation(
      SctpTransportState state,
      rtc::scoped_refptr<DtlsTransportInterface> dtls_transport,
      std::optional<double> max_message_size,
      std::optional<int> max_channels);
  ~SctpTransportInformation();
  // The DTLS transport that supports this SCTP transport.
  rtc::scoped_refptr<DtlsTransportInterface> dtls_transport() const {
    return dtls_transport_;
  }
  SctpTransportState state() const { return state_; }
  std::optional<double> MaxMessageSize() const { return max_message_size_; }
  std::optional<int> MaxChannels() const { return max_channels_; }

 private:
  SctpTransportState state_ = SctpTransportState::kNew;
  rtc::scoped_refptr<DtlsTransportInterface> dtls_transport_;
  std::optional<double> max_message_size_;
  std::optional<int> max_channels_;
};

class SctpTransportObserverInterface {
 public:
  // This callback carries information about the state of the transport.
  // The argument is a pass-by-value snapshot of the state.
  // The callback will be called on the network thread.
  virtual void OnStateChange(SctpTransportInformation info) = 0;

 protected:
  virtual ~SctpTransportObserverInterface() = default;
};

// A SCTP transport, as represented to the outside world.
// This object is created on the network thread, and can only be
// accessed on that thread, except for functions explicitly marked otherwise.
// References can be held by other threads, and destruction can therefore
// be initiated by other threads.
class SctpTransportInterface : public webrtc::RefCountInterface {
 public:
  // This function can be called from other threads.
  virtual rtc::scoped_refptr<DtlsTransportInterface> dtls_transport() const = 0;
  // Returns information on the state of the SctpTransport.
  // This function can be called from other threads.
  virtual SctpTransportInformation Information() const = 0;
  // Observer management.
  virtual void RegisterObserver(SctpTransportObserverInterface* observer) = 0;
  virtual void UnregisterObserver() = 0;
};

}  // namespace webrtc

#endif  // API_SCTP_TRANSPORT_INTERFACE_H_
