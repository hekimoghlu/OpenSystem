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
#ifndef _LINUX_IF_ALG_H
#define _LINUX_IF_ALG_H
#include <linux/types.h>
struct sockaddr_alg {
  __u16 salg_family;
  __u8 salg_type[14];
  __u32 salg_feat;
  __u32 salg_mask;
  __u8 salg_name[64];
};
struct sockaddr_alg_new {
  __u16 salg_family;
  __u8 salg_type[14];
  __u32 salg_feat;
  __u32 salg_mask;
  __u8 salg_name[];
};
struct af_alg_iv {
  __u32 ivlen;
  __u8 iv[];
};
#define ALG_SET_KEY 1
#define ALG_SET_IV 2
#define ALG_SET_OP 3
#define ALG_SET_AEAD_ASSOCLEN 4
#define ALG_SET_AEAD_AUTHSIZE 5
#define ALG_SET_DRBG_ENTROPY 6
#define ALG_SET_KEY_BY_KEY_SERIAL 7
#define ALG_OP_DECRYPT 0
#define ALG_OP_ENCRYPT 1
#endif
