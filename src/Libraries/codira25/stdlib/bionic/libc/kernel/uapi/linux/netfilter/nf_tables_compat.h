/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#ifndef _NFT_COMPAT_NFNETLINK_H_
#define _NFT_COMPAT_NFNETLINK_H_
enum nft_target_attributes {
  NFTA_TARGET_UNSPEC,
  NFTA_TARGET_NAME,
  NFTA_TARGET_REV,
  NFTA_TARGET_INFO,
  __NFTA_TARGET_MAX
};
#define NFTA_TARGET_MAX (__NFTA_TARGET_MAX - 1)
enum nft_match_attributes {
  NFTA_MATCH_UNSPEC,
  NFTA_MATCH_NAME,
  NFTA_MATCH_REV,
  NFTA_MATCH_INFO,
  __NFTA_MATCH_MAX
};
#define NFTA_MATCH_MAX (__NFTA_MATCH_MAX - 1)
#define NFT_COMPAT_NAME_MAX 32
enum {
  NFNL_MSG_COMPAT_GET,
  NFNL_MSG_COMPAT_MAX
};
enum {
  NFTA_COMPAT_UNSPEC = 0,
  NFTA_COMPAT_NAME,
  NFTA_COMPAT_REV,
  NFTA_COMPAT_TYPE,
  __NFTA_COMPAT_MAX,
};
#define NFTA_COMPAT_MAX (__NFTA_COMPAT_MAX - 1)
#endif
