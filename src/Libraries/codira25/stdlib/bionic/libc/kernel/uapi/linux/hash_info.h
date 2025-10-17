/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#ifndef _UAPI_LINUX_HASH_INFO_H
#define _UAPI_LINUX_HASH_INFO_H
enum hash_algo {
  HASH_ALGO_MD4,
  HASH_ALGO_MD5,
  HASH_ALGO_SHA1,
  HASH_ALGO_RIPE_MD_160,
  HASH_ALGO_SHA256,
  HASH_ALGO_SHA384,
  HASH_ALGO_SHA512,
  HASH_ALGO_SHA224,
  HASH_ALGO_RIPE_MD_128,
  HASH_ALGO_RIPE_MD_256,
  HASH_ALGO_RIPE_MD_320,
  HASH_ALGO_WP_256,
  HASH_ALGO_WP_384,
  HASH_ALGO_WP_512,
  HASH_ALGO_TGR_128,
  HASH_ALGO_TGR_160,
  HASH_ALGO_TGR_192,
  HASH_ALGO_SM3_256,
  HASH_ALGO_STREEBOG_256,
  HASH_ALGO_STREEBOG_512,
  HASH_ALGO_SHA3_256,
  HASH_ALGO_SHA3_384,
  HASH_ALGO_SHA3_512,
  HASH_ALGO__LAST
};
#endif
