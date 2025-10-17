/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#ifndef _UAPI_LINUX_MEMBARRIER_H
#define _UAPI_LINUX_MEMBARRIER_H
enum membarrier_cmd {
  MEMBARRIER_CMD_QUERY = 0,
  MEMBARRIER_CMD_GLOBAL = (1 << 0),
  MEMBARRIER_CMD_GLOBAL_EXPEDITED = (1 << 1),
  MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED = (1 << 2),
  MEMBARRIER_CMD_PRIVATE_EXPEDITED = (1 << 3),
  MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED = (1 << 4),
  MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE = (1 << 5),
  MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE = (1 << 6),
  MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ = (1 << 7),
  MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ = (1 << 8),
  MEMBARRIER_CMD_GET_REGISTRATIONS = (1 << 9),
  MEMBARRIER_CMD_SHARED = MEMBARRIER_CMD_GLOBAL,
};
enum membarrier_cmd_flag {
  MEMBARRIER_CMD_FLAG_CPU = (1 << 0),
};
#endif
