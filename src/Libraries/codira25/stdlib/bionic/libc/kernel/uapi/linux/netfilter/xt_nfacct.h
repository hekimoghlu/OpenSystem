/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#ifndef _XT_NFACCT_MATCH_H
#define _XT_NFACCT_MATCH_H
#include <linux/netfilter/nfnetlink_acct.h>
struct nf_acct;
struct xt_nfacct_match_info {
  char name[NFACCT_NAME_MAX];
  struct nf_acct * nfacct;
};
struct xt_nfacct_match_info_v1 {
  char name[NFACCT_NAME_MAX];
  struct nf_acct * nfacct __attribute__((aligned(8)));
};
#endif
