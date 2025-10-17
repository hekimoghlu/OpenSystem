/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include <netinet/ip_icmp.h>

#include <gtest/gtest.h>

TEST(netinet_ip_icmp, struct_icmphdr) {
  icmphdr hdr = { .type = ICMP_ECHO };
  ASSERT_EQ(ICMP_ECHO, hdr.type);
  ASSERT_EQ(0, hdr.code);
  ASSERT_EQ(0, hdr.checksum);
  ASSERT_EQ(0, hdr.un.echo.id);
  ASSERT_EQ(0, hdr.un.echo.sequence);
  ASSERT_EQ(0U, hdr.un.gateway);
  ASSERT_EQ(0, hdr.un.frag.mtu);
}
