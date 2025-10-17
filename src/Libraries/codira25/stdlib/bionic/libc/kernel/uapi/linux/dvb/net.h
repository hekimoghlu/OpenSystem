/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
#ifndef _DVBNET_H_
#define _DVBNET_H_
#include <linux/types.h>
struct dvb_net_if {
  __u16 pid;
  __u16 if_num;
  __u8 feedtype;
#define DVB_NET_FEEDTYPE_MPE 0
#define DVB_NET_FEEDTYPE_ULE 1
};
#define NET_ADD_IF _IOWR('o', 52, struct dvb_net_if)
#define NET_REMOVE_IF _IO('o', 53)
#define NET_GET_IF _IOWR('o', 54, struct dvb_net_if)
struct __dvb_net_if_old {
  __u16 pid;
  __u16 if_num;
};
#define __NET_ADD_IF_OLD _IOWR('o', 52, struct __dvb_net_if_old)
#define __NET_GET_IF_OLD _IOWR('o', 54, struct __dvb_net_if_old)
#endif
