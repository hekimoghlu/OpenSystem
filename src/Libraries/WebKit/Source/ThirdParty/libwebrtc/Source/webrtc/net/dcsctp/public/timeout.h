/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#ifndef NET_DCSCTP_PUBLIC_TIMEOUT_H_
#define NET_DCSCTP_PUBLIC_TIMEOUT_H_

#include <cstdint>

#include "net/dcsctp/public/types.h"

namespace dcsctp {

// A very simple timeout that can be started and stopped. When started,
// it will be given a unique `timeout_id` which should be provided to
// `DcSctpSocket::HandleTimeout` when it expires.
class Timeout {
 public:
  virtual ~Timeout() = default;

  // Called to start time timeout, with the duration in milliseconds as
  // `duration` and with the timeout identifier as `timeout_id`, which - if
  // the timeout expires - shall be provided to `DcSctpSocket::HandleTimeout`.
  //
  // `Start` and `Stop` will always be called in pairs. In other words will
  // Â´Start` never be called twice, without a call to `Stop` in between.
  virtual void Start(DurationMs duration, TimeoutID timeout_id) = 0;

  // Called to stop the running timeout.
  //
  // `Start` and `Stop` will always be called in pairs. In other words will
  // Â´Start` never be called twice, without a call to `Stop` in between.
  //
  // `Stop` will always be called prior to releasing this object.
  virtual void Stop() = 0;

  // Called to restart an already running timeout, with the `duration` and
  // `timeout_id` parameters as described in `Start`. This can be overridden by
  // the implementation to restart it more efficiently.
  virtual void Restart(DurationMs duration, TimeoutID timeout_id) {
    Stop();
    Start(duration, timeout_id);
  }
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PUBLIC_TIMEOUT_H_
