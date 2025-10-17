/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#ifndef _IP6T_REJECT_H
#define _IP6T_REJECT_H
#include <linux/types.h>
enum ip6t_reject_with {
  IP6T_ICMP6_NO_ROUTE,
  IP6T_ICMP6_ADM_PROHIBITED,
  IP6T_ICMP6_NOT_NEIGHBOUR,
  IP6T_ICMP6_ADDR_UNREACH,
  IP6T_ICMP6_PORT_UNREACH,
  IP6T_ICMP6_ECHOREPLY,
  IP6T_TCP_RESET,
  IP6T_ICMP6_POLICY_FAIL,
  IP6T_ICMP6_REJECT_ROUTE
};
struct ip6t_reject_info {
  __u32 with;
};
#endif
