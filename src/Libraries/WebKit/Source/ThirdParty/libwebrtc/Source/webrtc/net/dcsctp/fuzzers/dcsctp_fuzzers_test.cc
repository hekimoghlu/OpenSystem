/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

#include "api/array_view.h"
#include "net/dcsctp/packet/sctp_packet.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/socket/dcsctp_socket.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "rtc_base/logging.h"
#include "test/gmock.h"

namespace dcsctp {
namespace dcsctp_fuzzers {
namespace {

// This is a testbed where fuzzed data that cause issues can be evaluated and
// crashes reproduced. Use `xxd -i ./crash-abc` to generate `data` below.
TEST(DcsctpFuzzersTest, PassesTestbed) {
  uint8_t data[] = {0x07, 0x09, 0x00, 0x01, 0x11, 0xff, 0xff};

  FuzzerCallbacks cb;
  DcSctpOptions options;
  options.disable_checksum_verification = true;
  DcSctpSocket socket("A", cb, nullptr, options);

  FuzzSocket(socket, cb, data);
}

}  // namespace
}  // namespace dcsctp_fuzzers
}  // namespace dcsctp
