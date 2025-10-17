/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#ifndef _UAPI_LINUX_GTP_H_
#define _UAPI_LINUX_GTP_H_
#define GTP_GENL_MCGRP_NAME "gtp"
enum gtp_genl_cmds {
  GTP_CMD_NEWPDP,
  GTP_CMD_DELPDP,
  GTP_CMD_GETPDP,
  GTP_CMD_ECHOREQ,
  GTP_CMD_MAX,
};
enum gtp_version {
  GTP_V0 = 0,
  GTP_V1,
};
enum gtp_attrs {
  GTPA_UNSPEC = 0,
  GTPA_LINK,
  GTPA_VERSION,
  GTPA_TID,
  GTPA_PEER_ADDRESS,
#define GTPA_SGSN_ADDRESS GTPA_PEER_ADDRESS
  GTPA_MS_ADDRESS,
  GTPA_FLOW,
  GTPA_NET_NS_FD,
  GTPA_I_TEI,
  GTPA_O_TEI,
  GTPA_PAD,
  GTPA_PEER_ADDR6,
  GTPA_MS_ADDR6,
  GTPA_FAMILY,
  __GTPA_MAX,
};
#define GTPA_MAX (__GTPA_MAX - 1)
#endif
