/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#ifndef _UAPI_LINUX_FOU_H
#define _UAPI_LINUX_FOU_H
#define FOU_GENL_NAME "fou"
#define FOU_GENL_VERSION 1
enum {
  FOU_ENCAP_UNSPEC,
  FOU_ENCAP_DIRECT,
  FOU_ENCAP_GUE,
};
enum {
  FOU_ATTR_UNSPEC,
  FOU_ATTR_PORT,
  FOU_ATTR_AF,
  FOU_ATTR_IPPROTO,
  FOU_ATTR_TYPE,
  FOU_ATTR_REMCSUM_NOPARTIAL,
  FOU_ATTR_LOCAL_V4,
  FOU_ATTR_LOCAL_V6,
  FOU_ATTR_PEER_V4,
  FOU_ATTR_PEER_V6,
  FOU_ATTR_PEER_PORT,
  FOU_ATTR_IFINDEX,
  __FOU_ATTR_MAX
};
#define FOU_ATTR_MAX (__FOU_ATTR_MAX - 1)
enum {
  FOU_CMD_UNSPEC,
  FOU_CMD_ADD,
  FOU_CMD_DEL,
  FOU_CMD_GET,
  __FOU_CMD_MAX
};
#define FOU_CMD_MAX (__FOU_CMD_MAX - 1)
#endif
