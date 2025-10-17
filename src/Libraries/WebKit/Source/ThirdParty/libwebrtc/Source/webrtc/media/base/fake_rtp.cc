/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include "media/base/fake_rtp.h"

#include <stdint.h>
#include <string.h>

#include "absl/algorithm/container.h"
#include "rtc_base/checks.h"
#include "test/gtest.h"

void CompareHeaderExtensions(const char* packet1,
                             size_t packet1_size,
                             const char* packet2,
                             size_t packet2_size,
                             const std::vector<int>& encrypted_headers,
                             bool expect_equal) {
  // Sanity check: packets must be large enough to contain the RTP header and
  // extensions header.
  RTC_CHECK_GE(packet1_size, 12 + 4);
  RTC_CHECK_GE(packet2_size, 12 + 4);
  // RTP extension headers are the same.
  EXPECT_EQ(0, memcmp(packet1 + 12, packet2 + 12, 4));
  // Check for one-byte header extensions.
  EXPECT_EQ('\xBE', packet1[12]);
  EXPECT_EQ('\xDE', packet1[13]);
  // Determine position and size of extension headers.
  size_t extension_words = packet1[14] << 8 | packet1[15];
  const char* extension_data1 = packet1 + 12 + 4;
  const char* extension_end1 = extension_data1 + extension_words * 4;
  const char* extension_data2 = packet2 + 12 + 4;
  // Sanity check: packets must be large enough to contain the RTP header
  // extensions.
  RTC_CHECK_GE(packet1_size, 12 + 4 + extension_words * 4);
  RTC_CHECK_GE(packet2_size, 12 + 4 + extension_words * 4);
  while (extension_data1 < extension_end1) {
    uint8_t id = (*extension_data1 & 0xf0) >> 4;
    uint8_t len = (*extension_data1 & 0x0f) + 1;
    extension_data1++;
    extension_data2++;
    EXPECT_LE(extension_data1, extension_end1);
    if (id == 15) {
      // Finished parsing.
      break;
    }

    // The header extension doesn't get encrypted if the id is not in the
    // list of header extensions to encrypt.
    if (expect_equal || !absl::c_linear_search(encrypted_headers, id)) {
      EXPECT_EQ(0, memcmp(extension_data1, extension_data2, len));
    } else {
      EXPECT_NE(0, memcmp(extension_data1, extension_data2, len));
    }

    extension_data1 += len;
    extension_data2 += len;
    // Skip padding.
    while (extension_data1 < extension_end1 && *extension_data1 == 0) {
      extension_data1++;
      extension_data2++;
    }
  }
}
