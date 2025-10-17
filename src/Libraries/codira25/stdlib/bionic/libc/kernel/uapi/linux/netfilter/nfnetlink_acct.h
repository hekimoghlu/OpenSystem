/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#ifndef _UAPI_NFNL_ACCT_H_
#define _UAPI_NFNL_ACCT_H_
#ifndef NFACCT_NAME_MAX
#define NFACCT_NAME_MAX 32
#endif
enum nfnl_acct_msg_types {
  NFNL_MSG_ACCT_NEW,
  NFNL_MSG_ACCT_GET,
  NFNL_MSG_ACCT_GET_CTRZERO,
  NFNL_MSG_ACCT_DEL,
  NFNL_MSG_ACCT_OVERQUOTA,
  NFNL_MSG_ACCT_MAX
};
enum nfnl_acct_flags {
  NFACCT_F_QUOTA_PKTS = (1 << 0),
  NFACCT_F_QUOTA_BYTES = (1 << 1),
  NFACCT_F_OVERQUOTA = (1 << 2),
};
enum nfnl_acct_type {
  NFACCT_UNSPEC,
  NFACCT_NAME,
  NFACCT_PKTS,
  NFACCT_BYTES,
  NFACCT_USE,
  NFACCT_FLAGS,
  NFACCT_QUOTA,
  NFACCT_FILTER,
  NFACCT_PAD,
  __NFACCT_MAX
};
#define NFACCT_MAX (__NFACCT_MAX - 1)
enum nfnl_attr_filter_type {
  NFACCT_FILTER_UNSPEC,
  NFACCT_FILTER_MASK,
  NFACCT_FILTER_VALUE,
  __NFACCT_FILTER_MAX
};
#define NFACCT_FILTER_MAX (__NFACCT_FILTER_MAX - 1)
#endif
