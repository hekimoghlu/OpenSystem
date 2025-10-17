/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include "net/dcsctp/fuzzers/dcsctp_fuzzers.h"
#include "net/dcsctp/public/dcsctp_message.h"
#include "net/dcsctp/public/dcsctp_options.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/socket/dcsctp_socket.h"
#include "rtc_base/logging.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  dcsctp::dcsctp_fuzzers::FuzzerCallbacks cb;
  dcsctp::DcSctpOptions options;
  options.disable_checksum_verification = true;
  dcsctp::DcSctpSocket socket("A", cb, nullptr, options);

  dcsctp::dcsctp_fuzzers::FuzzSocket(socket, cb,
                                     rtc::ArrayView<const uint8_t>(data, size));
}
}  // namespace webrtc
