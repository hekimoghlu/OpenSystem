/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#ifndef _UAPI_LINUX_SEG6_HMAC_H
#define _UAPI_LINUX_SEG6_HMAC_H
#include <linux/types.h>
#include <linux/seg6.h>
#define SEG6_HMAC_SECRET_LEN 64
#define SEG6_HMAC_FIELD_LEN 32
struct sr6_tlv_hmac {
  struct sr6_tlv tlvhdr;
  __u16 reserved;
  __be32 hmackeyid;
  __u8 hmac[SEG6_HMAC_FIELD_LEN];
};
enum {
  SEG6_HMAC_ALGO_SHA1 = 1,
  SEG6_HMAC_ALGO_SHA256 = 2,
};
#endif
