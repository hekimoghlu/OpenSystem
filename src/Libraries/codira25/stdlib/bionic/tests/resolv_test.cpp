/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include <resolv.h>

#include <sys/cdefs.h>

#include <gtest/gtest.h>

TEST(resolv, b64_pton_28035006) {
  // Test data from https://groups.google.com/forum/#!topic/mailing.openbsd.tech/w3ACIlklJkI.
  const char* data =
      "p1v3+nehH3N3n+/OokzXpsyGF2VVpxIxkjSn3Mv/Sq74OE1iFuVU+K4bQImuVj"
      "S55RB2fpCpbB8Nye7tzrt6h9YPP3yyJfqORDETGmIB4lveZXA4KDxx50F9rYrO"
      "dFbTLyWfNBb/8Q2TnD72eY/3Y5P9qwtJwyDL25Tleic8G3g=";

  // This buffer is exactly the right size, but old versions of the BSD code
  // incorrectly required an extra byte. http://b/28035006.
  uint8_t buf[128];
  ASSERT_EQ(128, b64_pton(data, buf, sizeof(buf)));
}

TEST(resolv, b64_ntop) {
  char buf[128];
  memset(buf, 'x', sizeof(buf));
  ASSERT_EQ(static_cast<int>(strlen("aGVsbG8=")),
            b64_ntop(reinterpret_cast<u_char const*>("hello"), strlen("hello"),
                     buf, sizeof(buf)));
  ASSERT_STREQ(buf, "aGVsbG8=");
}

TEST(resolv, b64_pton) {
  u_char buf[128];
  memset(buf, 'x', sizeof(buf));
  ASSERT_EQ(static_cast<int>(strlen("hello")), b64_pton("aGVsbG8=", buf, sizeof(buf)));
  ASSERT_STREQ(reinterpret_cast<char*>(buf), "hello");
}

TEST(resolv, p_class) {
#if !defined(ANDROID_HOST_MUSL)
  ASSERT_STREQ("IN", p_class(ns_c_in));
  ASSERT_STREQ("BADCLASS", p_class(-1));
#else
  GTEST_SKIP() << "musl doesn't have p_class";
#endif
}

TEST(resolv, p_type) {
#if !defined(ANDROID_HOST_MUSL)
  ASSERT_STREQ("AAAA", p_type(ns_t_aaaa));
  ASSERT_STREQ("BADTYPE", p_type(-1));
#else
  GTEST_SKIP() << "musl doesn't have p_type";
#endif
}

TEST(resolv, res_init) {
  ASSERT_EQ(0, res_init());
}

TEST(resolv, res_randomid) {
#if !defined(ANDROID_HOST_MUSL)
  res_randomid();
#else
  GTEST_SKIP() << "musl doesn't have res_randomid";
#endif
}
