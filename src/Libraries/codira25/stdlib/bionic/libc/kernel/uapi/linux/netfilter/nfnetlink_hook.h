/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#ifndef _NFNL_HOOK_H_
#define _NFNL_HOOK_H_
enum nfnl_hook_msg_types {
  NFNL_MSG_HOOK_GET,
  NFNL_MSG_HOOK_MAX,
};
enum nfnl_hook_attributes {
  NFNLA_HOOK_UNSPEC,
  NFNLA_HOOK_HOOKNUM,
  NFNLA_HOOK_PRIORITY,
  NFNLA_HOOK_DEV,
  NFNLA_HOOK_FUNCTION_NAME,
  NFNLA_HOOK_MODULE_NAME,
  NFNLA_HOOK_CHAIN_INFO,
  __NFNLA_HOOK_MAX
};
#define NFNLA_HOOK_MAX (__NFNLA_HOOK_MAX - 1)
enum nfnl_hook_chain_info_attributes {
  NFNLA_HOOK_INFO_UNSPEC,
  NFNLA_HOOK_INFO_DESC,
  NFNLA_HOOK_INFO_TYPE,
  __NFNLA_HOOK_INFO_MAX,
};
#define NFNLA_HOOK_INFO_MAX (__NFNLA_HOOK_INFO_MAX - 1)
enum nfnl_hook_chain_desc_attributes {
  NFNLA_CHAIN_UNSPEC,
  NFNLA_CHAIN_TABLE,
  NFNLA_CHAIN_FAMILY,
  NFNLA_CHAIN_NAME,
  __NFNLA_CHAIN_MAX,
};
#define NFNLA_CHAIN_MAX (__NFNLA_CHAIN_MAX - 1)
enum nfnl_hook_chaintype {
  NFNL_HOOK_TYPE_NFTABLES = 0x1,
  NFNL_HOOK_TYPE_BPF,
};
enum nfnl_hook_bpf_attributes {
  NFNLA_HOOK_BPF_UNSPEC,
  NFNLA_HOOK_BPF_ID,
  __NFNLA_HOOK_BPF_MAX,
};
#define NFNLA_HOOK_BPF_MAX (__NFNLA_HOOK_BPF_MAX - 1)
#endif
