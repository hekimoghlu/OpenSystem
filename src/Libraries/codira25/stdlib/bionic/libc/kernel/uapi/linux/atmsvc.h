/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef _LINUX_ATMSVC_H
#define _LINUX_ATMSVC_H
#include <linux/atmapi.h>
#include <linux/atm.h>
#include <linux/atmioc.h>
#define ATMSIGD_CTRL _IO('a', ATMIOC_SPECIAL)
enum atmsvc_msg_type {
  as_catch_null,
  as_bind,
  as_connect,
  as_accept,
  as_reject,
  as_listen,
  as_okay,
  as_error,
  as_indicate,
  as_close,
  as_itf_notify,
  as_modify,
  as_identify,
  as_terminate,
  as_addparty,
  as_dropparty
};
struct atmsvc_msg {
  enum atmsvc_msg_type type;
  atm_kptr_t vcc;
  atm_kptr_t listen_vcc;
  int reply;
  struct sockaddr_atmpvc pvc;
  struct sockaddr_atmsvc local;
  struct atm_qos qos;
  struct atm_sap sap;
  unsigned int session;
  struct sockaddr_atmsvc svc;
} __ATM_API_ALIGN;
#define SELECT_TOP_PCR(tp) ((tp).pcr ? (tp).pcr : (tp).max_pcr && (tp).max_pcr != ATM_MAX_PCR ? (tp).max_pcr : (tp).min_pcr ? (tp).min_pcr : ATM_MAX_PCR)
#endif
