/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifdef WEBRTC_NETWORK_TESTER_TEST_ENABLED

#include <string>

#include "rtc_base/gunit.h"
#include "rtc_base/random.h"
#include "rtc_tools/network_tester/test_controller.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {

TEST(NetworkTesterTest, ServerClient) {
  // Use a unique port rather than a hard-coded one to avoid collision when
  // running the test in parallel in stress runs. Skipping all reserved ports.
  const int MIN_PORT = 49152;
  const int MAX_PORT = 65535;
  int port = webrtc::Random(rtc::TimeMicros()).Rand(MIN_PORT, MAX_PORT);

  rtc::AutoThread main_thread;

  TestController client(
      0, 0, webrtc::test::ResourcePath("network_tester/client_config", "dat"),
      webrtc::test::OutputPath() + "client_packet_log.dat");
  TestController server(
      port, port,
      webrtc::test::ResourcePath("network_tester/server_config", "dat"),
      webrtc::test::OutputPath() + "server_packet_log.dat");
  client.SendConnectTo("127.0.0.1", port);
  EXPECT_TRUE_WAIT(server.IsTestDone() && client.IsTestDone(), 2000);
}

}  // namespace webrtc

#endif  // WEBRTC_NETWORK_TESTER_TEST_ENABLED
