/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#ifndef __LINUX_TC_TUNNEL_KEY_H
#define __LINUX_TC_TUNNEL_KEY_H
#include <linux/pkt_cls.h>
#define TCA_TUNNEL_KEY_ACT_SET 1
#define TCA_TUNNEL_KEY_ACT_RELEASE 2
struct tc_tunnel_key {
  tc_gen;
  int t_action;
};
enum {
  TCA_TUNNEL_KEY_UNSPEC,
  TCA_TUNNEL_KEY_TM,
  TCA_TUNNEL_KEY_PARMS,
  TCA_TUNNEL_KEY_ENC_IPV4_SRC,
  TCA_TUNNEL_KEY_ENC_IPV4_DST,
  TCA_TUNNEL_KEY_ENC_IPV6_SRC,
  TCA_TUNNEL_KEY_ENC_IPV6_DST,
  TCA_TUNNEL_KEY_ENC_KEY_ID,
  TCA_TUNNEL_KEY_PAD,
  TCA_TUNNEL_KEY_ENC_DST_PORT,
  TCA_TUNNEL_KEY_NO_CSUM,
  TCA_TUNNEL_KEY_ENC_OPTS,
  TCA_TUNNEL_KEY_ENC_TOS,
  TCA_TUNNEL_KEY_ENC_TTL,
  TCA_TUNNEL_KEY_NO_FRAG,
  __TCA_TUNNEL_KEY_MAX,
};
#define TCA_TUNNEL_KEY_MAX (__TCA_TUNNEL_KEY_MAX - 1)
enum {
  TCA_TUNNEL_KEY_ENC_OPTS_UNSPEC,
  TCA_TUNNEL_KEY_ENC_OPTS_GENEVE,
  TCA_TUNNEL_KEY_ENC_OPTS_VXLAN,
  TCA_TUNNEL_KEY_ENC_OPTS_ERSPAN,
  __TCA_TUNNEL_KEY_ENC_OPTS_MAX,
};
#define TCA_TUNNEL_KEY_ENC_OPTS_MAX (__TCA_TUNNEL_KEY_ENC_OPTS_MAX - 1)
enum {
  TCA_TUNNEL_KEY_ENC_OPT_GENEVE_UNSPEC,
  TCA_TUNNEL_KEY_ENC_OPT_GENEVE_CLASS,
  TCA_TUNNEL_KEY_ENC_OPT_GENEVE_TYPE,
  TCA_TUNNEL_KEY_ENC_OPT_GENEVE_DATA,
  __TCA_TUNNEL_KEY_ENC_OPT_GENEVE_MAX,
};
#define TCA_TUNNEL_KEY_ENC_OPT_GENEVE_MAX (__TCA_TUNNEL_KEY_ENC_OPT_GENEVE_MAX - 1)
enum {
  TCA_TUNNEL_KEY_ENC_OPT_VXLAN_UNSPEC,
  TCA_TUNNEL_KEY_ENC_OPT_VXLAN_GBP,
  __TCA_TUNNEL_KEY_ENC_OPT_VXLAN_MAX,
};
#define TCA_TUNNEL_KEY_ENC_OPT_VXLAN_MAX (__TCA_TUNNEL_KEY_ENC_OPT_VXLAN_MAX - 1)
enum {
  TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_UNSPEC,
  TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_VER,
  TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_INDEX,
  TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_DIR,
  TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_HWID,
  __TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_MAX,
};
#define TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_MAX (__TCA_TUNNEL_KEY_ENC_OPT_ERSPAN_MAX - 1)
#endif
