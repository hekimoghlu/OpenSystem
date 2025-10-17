/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#ifndef PC_SCTP_TRANSPORT_H_
#define PC_SCTP_TRANSPORT_H_

#include <memory>

#include "api/dtls_transport_interface.h"
#include "api/scoped_refptr.h"
#include "api/sctp_transport_interface.h"
#include "api/sequence_checker.h"
#include "api/transport/data_channel_transport_interface.h"
#include "media/sctp/sctp_transport_internal.h"
#include "p2p/base/dtls_transport_internal.h"
#include "pc/dtls_transport.h"
#include "rtc_base/checks.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// This implementation wraps a cricket::SctpTransport, and takes
// ownership of it.
// This object must be constructed and updated on the networking thread,
// the same thread as the one the cricket::SctpTransportInternal object
// lives on.
class SctpTransport : public SctpTransportInterface,
                      public DataChannelTransportInterface {
 public:
  SctpTransport(std::unique_ptr<cricket::SctpTransportInternal> internal,
                rtc::scoped_refptr<DtlsTransport> dtls_transport);

  // SctpTransportInterface
  rtc::scoped_refptr<DtlsTransportInterface> dtls_transport() const override;
  SctpTransportInformation Information() const override;
  void RegisterObserver(SctpTransportObserverInterface* observer) override;
  void UnregisterObserver() override;

  // DataChannelTransportInterface
  RTCError OpenChannel(int channel_id, PriorityValue priority) override;
  RTCError SendData(int channel_id,
                    const SendDataParams& params,
                    const rtc::CopyOnWriteBuffer& buffer) override;
  RTCError CloseChannel(int channel_id) override;
  void SetDataSink(DataChannelSink* sink) override;
  bool IsReadyToSend() const override;
  size_t buffered_amount(int channel_id) const override;
  size_t buffered_amount_low_threshold(int channel_id) const override;
  void SetBufferedAmountLowThreshold(int channel_id, size_t bytes) override;

  // Internal functions
  void Clear();
  // Initialize the cricket::SctpTransport. This can be called from
  // the signaling thread.
  void Start(int local_port, int remote_port, int max_message_size);

  // TODO(https://bugs.webrtc.org/10629): Move functions that need
  // internal() to be functions on the SctpTransport interface,
  // and make the internal() function private.
  cricket::SctpTransportInternal* internal() {
    RTC_DCHECK_RUN_ON(owner_thread_);
    return internal_sctp_transport_.get();
  }

  const cricket::SctpTransportInternal* internal() const {
    RTC_DCHECK_RUN_ON(owner_thread_);
    return internal_sctp_transport_.get();
  }

 protected:
  ~SctpTransport() override;

 private:
  void UpdateInformation(SctpTransportState state);
  void OnInternalReadyToSendData();
  void OnAssociationChangeCommunicationUp();
  void OnInternalClosingProcedureStartedRemotely(int sid);
  void OnInternalClosingProcedureComplete(int sid);
  void OnDtlsStateChange(cricket::DtlsTransportInternal* transport,
                         DtlsTransportState state);

  // NOTE: `owner_thread_` is the thread that the SctpTransport object is
  // constructed on. In the context of PeerConnection, it's the network thread.
  rtc::Thread* const owner_thread_;
  SctpTransportInformation info_ RTC_GUARDED_BY(owner_thread_);
  std::unique_ptr<cricket::SctpTransportInternal> internal_sctp_transport_
      RTC_GUARDED_BY(owner_thread_);
  SctpTransportObserverInterface* observer_ RTC_GUARDED_BY(owner_thread_) =
      nullptr;
  rtc::scoped_refptr<DtlsTransport> dtls_transport_
      RTC_GUARDED_BY(owner_thread_);
};

}  // namespace webrtc
#endif  // PC_SCTP_TRANSPORT_H_
