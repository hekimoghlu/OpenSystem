/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef _UAPI_NF_CONNTRACK_SCTP_H
#define _UAPI_NF_CONNTRACK_SCTP_H
#include <linux/netfilter/nf_conntrack_tuple_common.h>
enum sctp_conntrack {
  SCTP_CONNTRACK_NONE,
  SCTP_CONNTRACK_CLOSED,
  SCTP_CONNTRACK_COOKIE_WAIT,
  SCTP_CONNTRACK_COOKIE_ECHOED,
  SCTP_CONNTRACK_ESTABLISHED,
  SCTP_CONNTRACK_SHUTDOWN_SENT,
  SCTP_CONNTRACK_SHUTDOWN_RECD,
  SCTP_CONNTRACK_SHUTDOWN_ACK_SENT,
  SCTP_CONNTRACK_HEARTBEAT_SENT,
  SCTP_CONNTRACK_HEARTBEAT_ACKED,
  SCTP_CONNTRACK_MAX
};
#endif
