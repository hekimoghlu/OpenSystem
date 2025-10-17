/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include "rtc_base/null_socket_server.h"
#include "rtc_base/thread.h"
#include "rtc_tools/network_tester/test_controller.h"

int main(int /*argn*/, char* /*argv*/[]) {
  rtc::Thread main_thread(std::make_unique<rtc::NullSocketServer>());
  webrtc::TestController server(9090, 9090, "server_config.dat",
                                "server_packet_log.dat");
  while (!server.IsTestDone()) {
    // 100 ms is arbitrary chosen.
    main_thread.ProcessMessages(/*cms=*/100);
  }
  return 0;
}
