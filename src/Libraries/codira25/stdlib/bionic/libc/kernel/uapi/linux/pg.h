/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#ifndef _UAPI_LINUX_PG_H
#define _UAPI_LINUX_PG_H
#define PG_MAGIC 'P'
#define PG_RESET 'Z'
#define PG_COMMAND 'C'
#define PG_MAX_DATA 32768
struct pg_write_hdr {
  char magic;
  char func;
  int dlen;
  int timeout;
  char packet[12];
};
struct pg_read_hdr {
  char magic;
  char scsi;
  int dlen;
  int duration;
  char pad[12];
};
#endif
