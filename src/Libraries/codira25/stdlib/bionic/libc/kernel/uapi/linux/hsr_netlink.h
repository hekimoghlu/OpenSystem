/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#ifndef __UAPI_HSR_NETLINK_H
#define __UAPI_HSR_NETLINK_H
enum {
  HSR_A_UNSPEC,
  HSR_A_NODE_ADDR,
  HSR_A_IFINDEX,
  HSR_A_IF1_AGE,
  HSR_A_IF2_AGE,
  HSR_A_NODE_ADDR_B,
  HSR_A_IF1_SEQ,
  HSR_A_IF2_SEQ,
  HSR_A_IF1_IFINDEX,
  HSR_A_IF2_IFINDEX,
  HSR_A_ADDR_B_IFINDEX,
  __HSR_A_MAX,
};
#define HSR_A_MAX (__HSR_A_MAX - 1)
enum {
  HSR_C_UNSPEC,
  HSR_C_RING_ERROR,
  HSR_C_NODE_DOWN,
  HSR_C_GET_NODE_STATUS,
  HSR_C_SET_NODE_STATUS,
  HSR_C_GET_NODE_LIST,
  HSR_C_SET_NODE_LIST,
  __HSR_C_MAX,
};
#define HSR_C_MAX (__HSR_C_MAX - 1)
#endif
