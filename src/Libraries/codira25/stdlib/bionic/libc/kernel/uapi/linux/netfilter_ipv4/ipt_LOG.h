/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#ifndef _IPT_LOG_H
#define _IPT_LOG_H
#define IPT_LOG_TCPSEQ 0x01
#define IPT_LOG_TCPOPT 0x02
#define IPT_LOG_IPOPT 0x04
#define IPT_LOG_UID 0x08
#define IPT_LOG_NFLOG 0x10
#define IPT_LOG_MACDECODE 0x20
#define IPT_LOG_MASK 0x2f
struct ipt_log_info {
  unsigned char level;
  unsigned char logflags;
  char prefix[30];
};
#endif
