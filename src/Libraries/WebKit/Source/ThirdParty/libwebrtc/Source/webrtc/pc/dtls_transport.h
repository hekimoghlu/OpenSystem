/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef PC_DTLS_TRANSPORT_H_
#define PC_DTLS_TRANSPORT_H_

#include <memory>
#include <utility>

#include "api/dtls_transport_interface.h"
#include "api/ice_transport_interface.h"
#include "api/scoped_refptr.h"
#include "p2p/base/dtls_transport.h"
#include "p2p/base/dtls_transport_internal.h"
#include "pc/ice_transport.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class IceTransportWithPointer;

// This implementation wraps a cricket::DtlsTransport, and takes
// ownership of it.
class DtlsTransport : public DtlsTransportInterface {
 public:
  // This object must be constructed and updated on a consistent thread,
  // the same thread as the one the cricket::DtlsTransportInternal object
  // lives on.
  // The Information() function can be called from a different thread,
  // such as the signalling thread.
  explicit DtlsTransport(
      std::unique_ptr<cricket::DtlsTransportInternal> internal);

  rtc::scoped_refptr<IceTransportInterface> ice_transport() override;

  // Currently called from the signaling thread and potentially Chromium's
  // JS thread.
  DtlsTransportInformation Information() override;

  void RegisterObserver(DtlsTransportObserverInterface* observer) override;
  void UnregisterObserver() override;
  void Clear();

  cricket::DtlsTransportInternal* internal() {
    RTC_DCHECK_RUN_ON(owner_thread_);
    return internal_dtls_transport_.get();
  }

  const cricket::DtlsTransportInternal* internal() const {
    RTC_DCHECK_RUN_ON(owner_thread_);
    return internal_dtls_transport_.get();
  }

 protected:
  ~DtlsTransport();

 private:
  void OnInternalDtlsState(cricket::DtlsTransportInternal* transport,
                           DtlsTransportState state);
  void UpdateInformation();

  // Called when changing `info_`. We only change the values from the
  // `owner_thread_` (a.k.a. the network thread).
  void set_info(DtlsTransportInformation&& info) RTC_RUN_ON(owner_thread_) {
    MutexLock lock(&lock_);
    info_ = std::move(info);
  }

  DtlsTransportObserverInterface* observer_ = nullptr;
  rtc::Thread* owner_thread_;
  mutable Mutex lock_;
  DtlsTransportInformation info_ RTC_GUARDED_BY(lock_);
  std::unique_ptr<cricket::DtlsTransportInternal> internal_dtls_transport_
      RTC_GUARDED_BY(owner_thread_);
  const rtc::scoped_refptr<IceTransportWithPointer> ice_transport_;
};

}  // namespace webrtc
#endif  // PC_DTLS_TRANSPORT_H_
