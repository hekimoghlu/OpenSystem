/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#ifndef _UAPILINUX_ATM_TCP_H
#define _UAPILINUX_ATM_TCP_H
#include <linux/atmapi.h>
#include <linux/atm.h>
#include <linux/atmioc.h>
#include <linux/types.h>
struct atmtcp_hdr {
  __u16 vpi;
  __u16 vci;
  __u32 length;
};
#define ATMTCP_HDR_MAGIC (~0)
#define ATMTCP_CTRL_OPEN 1
#define ATMTCP_CTRL_CLOSE 2
struct atmtcp_control {
  struct atmtcp_hdr hdr;
  int type;
  atm_kptr_t vcc;
  struct sockaddr_atmpvc addr;
  struct atm_qos qos;
  int result;
} __ATM_API_ALIGN;
#define SIOCSIFATMTCP _IO('a', ATMIOC_ITF)
#define ATMTCP_CREATE _IO('a', ATMIOC_ITF + 14)
#define ATMTCP_REMOVE _IO('a', ATMIOC_ITF + 15)
#endif
