/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#ifndef _UAPI_LINUX_ILA_H
#define _UAPI_LINUX_ILA_H
#define ILA_GENL_NAME "ila"
#define ILA_GENL_VERSION 0x1
enum {
  ILA_ATTR_UNSPEC,
  ILA_ATTR_LOCATOR,
  ILA_ATTR_IDENTIFIER,
  ILA_ATTR_LOCATOR_MATCH,
  ILA_ATTR_IFINDEX,
  ILA_ATTR_DIR,
  ILA_ATTR_PAD,
  ILA_ATTR_CSUM_MODE,
  ILA_ATTR_IDENT_TYPE,
  ILA_ATTR_HOOK_TYPE,
  __ILA_ATTR_MAX,
};
#define ILA_ATTR_MAX (__ILA_ATTR_MAX - 1)
enum {
  ILA_CMD_UNSPEC,
  ILA_CMD_ADD,
  ILA_CMD_DEL,
  ILA_CMD_GET,
  ILA_CMD_FLUSH,
  __ILA_CMD_MAX,
};
#define ILA_CMD_MAX (__ILA_CMD_MAX - 1)
#define ILA_DIR_IN (1 << 0)
#define ILA_DIR_OUT (1 << 1)
enum {
  ILA_CSUM_ADJUST_TRANSPORT,
  ILA_CSUM_NEUTRAL_MAP,
  ILA_CSUM_NO_ACTION,
  ILA_CSUM_NEUTRAL_MAP_AUTO,
};
enum {
  ILA_ATYPE_IID = 0,
  ILA_ATYPE_LUID,
  ILA_ATYPE_VIRT_V4,
  ILA_ATYPE_VIRT_UNI_V6,
  ILA_ATYPE_VIRT_MULTI_V6,
  ILA_ATYPE_NONLOCAL_ADDR,
  ILA_ATYPE_RSVD_1,
  ILA_ATYPE_RSVD_2,
  ILA_ATYPE_USE_FORMAT = 32,
};
enum {
  ILA_HOOK_ROUTE_OUTPUT,
  ILA_HOOK_ROUTE_INPUT,
};
#endif
