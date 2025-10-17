/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef PC_ICE_TRANSPORT_H_
#define PC_ICE_TRANSPORT_H_

#include "api/ice_transport_interface.h"
#include "rtc_base/checks.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// Implementation of IceTransportInterface that does not take ownership
// of its underlying IceTransport. It depends on its creator class to
// ensure that Clear() is called before the underlying IceTransport
// is deallocated.
class IceTransportWithPointer : public IceTransportInterface {
 public:
  explicit IceTransportWithPointer(cricket::IceTransportInternal* internal)
      : creator_thread_(rtc::Thread::Current()), internal_(internal) {
    RTC_DCHECK(internal_);
  }

  IceTransportWithPointer() = delete;
  IceTransportWithPointer(const IceTransportWithPointer&) = delete;
  IceTransportWithPointer& operator=(const IceTransportWithPointer&) = delete;

  cricket::IceTransportInternal* internal() override;
  // This call will ensure that the pointer passed at construction is
  // no longer in use by this object. Later calls to internal() will return
  // null.
  void Clear();

 protected:
  ~IceTransportWithPointer() override;

 private:
  const rtc::Thread* creator_thread_;
  cricket::IceTransportInternal* internal_ RTC_GUARDED_BY(creator_thread_);
};

}  // namespace webrtc

#endif  // PC_ICE_TRANSPORT_H_
