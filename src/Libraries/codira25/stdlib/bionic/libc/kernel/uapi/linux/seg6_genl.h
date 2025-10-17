/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#ifndef _UAPI_LINUX_SEG6_GENL_H
#define _UAPI_LINUX_SEG6_GENL_H
#define SEG6_GENL_NAME "SEG6"
#define SEG6_GENL_VERSION 0x1
enum {
  SEG6_ATTR_UNSPEC,
  SEG6_ATTR_DST,
  SEG6_ATTR_DSTLEN,
  SEG6_ATTR_HMACKEYID,
  SEG6_ATTR_SECRET,
  SEG6_ATTR_SECRETLEN,
  SEG6_ATTR_ALGID,
  SEG6_ATTR_HMACINFO,
  __SEG6_ATTR_MAX,
};
#define SEG6_ATTR_MAX (__SEG6_ATTR_MAX - 1)
enum {
  SEG6_CMD_UNSPEC,
  SEG6_CMD_SETHMAC,
  SEG6_CMD_DUMPHMAC,
  SEG6_CMD_SET_TUNSRC,
  SEG6_CMD_GET_TUNSRC,
  __SEG6_CMD_MAX,
};
#define SEG6_CMD_MAX (__SEG6_CMD_MAX - 1)
#endif
