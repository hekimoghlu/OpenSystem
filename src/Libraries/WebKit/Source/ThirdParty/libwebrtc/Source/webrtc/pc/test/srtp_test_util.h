/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#ifndef PC_TEST_SRTP_TEST_UTIL_H_
#define PC_TEST_SRTP_TEST_UTIL_H_

#include "rtc_base/ssl_stream_adapter.h"

namespace rtc {

static const rtc::ZeroOnFreeBuffer<uint8_t> kTestKey1{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234", 30};
static const rtc::ZeroOnFreeBuffer<uint8_t> kTestKey2{
    "4321ZYXWVUTSRQPONMLKJIHGFEDCBA", 30};

static int rtp_auth_tag_len(int crypto_suite) {
  switch (crypto_suite) {
    case kSrtpAes128CmSha1_32:
      return 4;
    case kSrtpAes128CmSha1_80:
      return 10;
    case kSrtpAeadAes128Gcm:
    case kSrtpAeadAes256Gcm:
      return 16;
    default:
      RTC_CHECK_NOTREACHED();
  }
}

static int rtcp_auth_tag_len(int crypto_suite) {
  switch (crypto_suite) {
    case kSrtpAes128CmSha1_32:
    case kSrtpAes128CmSha1_80:
      return 10;
    case kSrtpAeadAes128Gcm:
    case kSrtpAeadAes256Gcm:
      return 16;
    default:
      RTC_CHECK_NOTREACHED();
  }
}

}  // namespace rtc

#endif  // PC_TEST_SRTP_TEST_UTIL_H_
