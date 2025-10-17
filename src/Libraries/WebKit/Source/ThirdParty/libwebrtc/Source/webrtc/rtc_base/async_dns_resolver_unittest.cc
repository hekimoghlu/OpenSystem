/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#include "rtc_base/async_dns_resolver.h"

#include "rtc_base/gunit.h"
#include "test/gtest.h"
#include "test/run_loop.h"

namespace webrtc {
namespace {
const int kDefaultTimeout = 1000;
const int kPortNumber = 3027;

TEST(AsyncDnsResolver, ConstructorWorks) {
  AsyncDnsResolver resolver;
}

TEST(AsyncDnsResolver, ResolvingLocalhostWorks) {
  test::RunLoop loop;  // Ensure that posting back to main thread works
  AsyncDnsResolver resolver;
  rtc::SocketAddress address("localhost",
                             kPortNumber);  // Port number does not matter
  rtc::SocketAddress resolved_address;
  bool done = false;
  resolver.Start(address, [&done] { done = true; });
  ASSERT_TRUE_WAIT(done, kDefaultTimeout);
  EXPECT_EQ(resolver.result().GetError(), 0);
  if (resolver.result().GetResolvedAddress(AF_INET, &resolved_address)) {
    EXPECT_EQ(resolved_address, rtc::SocketAddress("127.0.0.1", kPortNumber));
  } else {
    RTC_LOG(LS_INFO) << "Resolution gave no address, skipping test";
  }
}

TEST(AsyncDnsResolver, ResolveAfterDeleteDoesNotReturn) {
  test::RunLoop loop;
  std::unique_ptr<AsyncDnsResolver> resolver =
      std::make_unique<AsyncDnsResolver>();
  rtc::SocketAddress address("localhost",
                             kPortNumber);  // Port number does not matter
  rtc::SocketAddress resolved_address;
  bool done = false;
  resolver->Start(address, [&done] { done = true; });
  resolver.reset();                    // Deletes resolver.
  rtc::Thread::Current()->SleepMs(1);  // Allows callback to execute
  EXPECT_FALSE(done);                  // Expect no result.
}

}  // namespace
}  // namespace webrtc
