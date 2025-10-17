/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#include "pc/dtls_transport.h"

#include <optional>
#include <utility>

#include "api/dtls_transport_interface.h"
#include "api/make_ref_counted.h"
#include "api/sequence_checker.h"
#include "pc/ice_transport.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/ssl_stream_adapter.h"

namespace webrtc {

// Implementation of DtlsTransportInterface
DtlsTransport::DtlsTransport(
    std::unique_ptr<cricket::DtlsTransportInternal> internal)
    : owner_thread_(rtc::Thread::Current()),
      info_(DtlsTransportState::kNew),
      internal_dtls_transport_(std::move(internal)),
      ice_transport_(rtc::make_ref_counted<IceTransportWithPointer>(
          internal_dtls_transport_->ice_transport())) {
  RTC_DCHECK(internal_dtls_transport_.get());
  internal_dtls_transport_->SubscribeDtlsTransportState(
      [this](cricket::DtlsTransportInternal* transport,
             DtlsTransportState state) {
        OnInternalDtlsState(transport, state);
      });
  UpdateInformation();
}

DtlsTransport::~DtlsTransport() {
  // TODO(tommi): Due to a reference being held by the RtpSenderBase
  // implementation, the last reference to the `DtlsTransport` instance can
  // be released on the signaling thread.
  // RTC_DCHECK_RUN_ON(owner_thread_);

  // We depend on the signaling thread to call Clear() before dropping
  // its last reference to this object.

  // If there are non `owner_thread_` references outstanding, and those
  // references are the last ones released, we depend on Clear() having been
  // called from the owner_thread before the last reference is deleted.
  // `Clear()` is currently called from `JsepTransport::~JsepTransport`.
  RTC_DCHECK(owner_thread_->IsCurrent() || !internal_dtls_transport_);
}

DtlsTransportInformation DtlsTransport::Information() {
  MutexLock lock(&lock_);
  return info_;
}

void DtlsTransport::RegisterObserver(DtlsTransportObserverInterface* observer) {
  RTC_DCHECK_RUN_ON(owner_thread_);
  RTC_DCHECK(observer);
  observer_ = observer;
}

void DtlsTransport::UnregisterObserver() {
  RTC_DCHECK_RUN_ON(owner_thread_);
  observer_ = nullptr;
}

rtc::scoped_refptr<IceTransportInterface> DtlsTransport::ice_transport() {
  return ice_transport_;
}

// Internal functions
void DtlsTransport::Clear() {
  RTC_DCHECK_RUN_ON(owner_thread_);
  RTC_DCHECK(internal());
  bool must_send_event =
      (internal()->dtls_state() != DtlsTransportState::kClosed);
  internal_dtls_transport_.reset();
  ice_transport_->Clear();
  UpdateInformation();
  if (observer_ && must_send_event) {
    observer_->OnStateChange(Information());
  }
}

void DtlsTransport::OnInternalDtlsState(
    cricket::DtlsTransportInternal* transport,
    DtlsTransportState state) {
  RTC_DCHECK_RUN_ON(owner_thread_);
  RTC_DCHECK(transport == internal());
  RTC_DCHECK(state == internal()->dtls_state());
  UpdateInformation();
  if (observer_) {
    observer_->OnStateChange(Information());
  }
}

void DtlsTransport::UpdateInformation() {
  RTC_DCHECK_RUN_ON(owner_thread_);
  if (internal_dtls_transport_) {
    if (internal_dtls_transport_->dtls_state() ==
        DtlsTransportState::kConnected) {
      bool success = true;
      rtc::SSLRole internal_role;
      std::optional<DtlsTransportTlsRole> role;
      int ssl_cipher_suite;
      int tls_version;
      int srtp_cipher;
      success &= internal_dtls_transport_->GetDtlsRole(&internal_role);
      if (success) {
        switch (internal_role) {
          case rtc::SSL_CLIENT:
            role = DtlsTransportTlsRole::kClient;
            break;
          case rtc::SSL_SERVER:
            role = DtlsTransportTlsRole::kServer;
            break;
        }
      }
      success &= internal_dtls_transport_->GetSslVersionBytes(&tls_version);
      success &= internal_dtls_transport_->GetSslCipherSuite(&ssl_cipher_suite);
      success &= internal_dtls_transport_->GetSrtpCryptoSuite(&srtp_cipher);
      if (success) {
        set_info(DtlsTransportInformation(
            internal_dtls_transport_->dtls_state(), role, tls_version,
            ssl_cipher_suite, srtp_cipher,
            internal_dtls_transport_->GetRemoteSSLCertChain()));
      } else {
        RTC_LOG(LS_ERROR) << "DtlsTransport in connected state has incomplete "
                             "TLS information";
        set_info(DtlsTransportInformation(
            internal_dtls_transport_->dtls_state(), role, std::nullopt,
            std::nullopt, std::nullopt,
            internal_dtls_transport_->GetRemoteSSLCertChain()));
      }
    } else {
      set_info(
          DtlsTransportInformation(internal_dtls_transport_->dtls_state()));
    }
  } else {
    set_info(DtlsTransportInformation(DtlsTransportState::kClosed));
  }
}

}  // namespace webrtc
