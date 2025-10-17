/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#ifndef _UAPI_LINUX_IF_EQL_H
#define _UAPI_LINUX_IF_EQL_H
#define EQL_DEFAULT_SLAVE_PRIORITY 28800
#define EQL_DEFAULT_MAX_SLAVES 4
#define EQL_DEFAULT_MTU 576
#define EQL_DEFAULT_RESCHED_IVAL HZ
#define EQL_ENSLAVE (SIOCDEVPRIVATE)
#define EQL_EMANCIPATE (SIOCDEVPRIVATE + 1)
#define EQL_GETSLAVECFG (SIOCDEVPRIVATE + 2)
#define EQL_SETSLAVECFG (SIOCDEVPRIVATE + 3)
#define EQL_GETMASTRCFG (SIOCDEVPRIVATE + 4)
#define EQL_SETMASTRCFG (SIOCDEVPRIVATE + 5)
typedef struct master_config {
  char master_name[16];
  int max_slaves;
  int min_slaves;
} master_config_t;
typedef struct slave_config {
  char slave_name[16];
  long priority;
} slave_config_t;
typedef struct slaving_request {
  char slave_name[16];
  long priority;
} slaving_request_t;
#endif
