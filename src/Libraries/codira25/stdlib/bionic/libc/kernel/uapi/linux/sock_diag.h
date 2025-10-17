/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#ifndef _UAPI__SOCK_DIAG_H__
#define _UAPI__SOCK_DIAG_H__
#include <linux/types.h>
#define SOCK_DIAG_BY_FAMILY 20
#define SOCK_DESTROY 21
struct sock_diag_req {
  __u8 sdiag_family;
  __u8 sdiag_protocol;
};
enum {
  SK_MEMINFO_RMEM_ALLOC,
  SK_MEMINFO_RCVBUF,
  SK_MEMINFO_WMEM_ALLOC,
  SK_MEMINFO_SNDBUF,
  SK_MEMINFO_FWD_ALLOC,
  SK_MEMINFO_WMEM_QUEUED,
  SK_MEMINFO_OPTMEM,
  SK_MEMINFO_BACKLOG,
  SK_MEMINFO_DROPS,
  SK_MEMINFO_VARS,
};
enum sknetlink_groups {
  SKNLGRP_NONE,
  SKNLGRP_INET_TCP_DESTROY,
  SKNLGRP_INET_UDP_DESTROY,
  SKNLGRP_INET6_TCP_DESTROY,
  SKNLGRP_INET6_UDP_DESTROY,
  __SKNLGRP_MAX,
};
#define SKNLGRP_MAX (__SKNLGRP_MAX - 1)
enum {
  SK_DIAG_BPF_STORAGE_REQ_NONE,
  SK_DIAG_BPF_STORAGE_REQ_MAP_FD,
  __SK_DIAG_BPF_STORAGE_REQ_MAX,
};
#define SK_DIAG_BPF_STORAGE_REQ_MAX (__SK_DIAG_BPF_STORAGE_REQ_MAX - 1)
enum {
  SK_DIAG_BPF_STORAGE_REP_NONE,
  SK_DIAG_BPF_STORAGE,
  __SK_DIAG_BPF_STORAGE_REP_MAX,
};
#define SK_DIAB_BPF_STORAGE_REP_MAX (__SK_DIAG_BPF_STORAGE_REP_MAX - 1)
enum {
  SK_DIAG_BPF_STORAGE_NONE,
  SK_DIAG_BPF_STORAGE_PAD,
  SK_DIAG_BPF_STORAGE_MAP_ID,
  SK_DIAG_BPF_STORAGE_MAP_VALUE,
  __SK_DIAG_BPF_STORAGE_MAX,
};
#define SK_DIAG_BPF_STORAGE_MAX (__SK_DIAG_BPF_STORAGE_MAX - 1)
#endif
