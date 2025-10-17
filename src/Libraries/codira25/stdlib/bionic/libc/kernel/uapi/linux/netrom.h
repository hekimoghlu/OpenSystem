/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef NETROM_KERNEL_H
#define NETROM_KERNEL_H
#include <linux/ax25.h>
#define NETROM_MTU 236
#define NETROM_T1 1
#define NETROM_T2 2
#define NETROM_N2 3
#define NETROM_T4 6
#define NETROM_IDLE 7
#define SIOCNRDECOBS (SIOCPROTOPRIVATE + 2)
struct nr_route_struct {
#define NETROM_NEIGH 0
#define NETROM_NODE 1
  int type;
  ax25_address callsign;
  char device[16];
  unsigned int quality;
  char mnemonic[7];
  ax25_address neighbour;
  unsigned int obs_count;
  unsigned int ndigis;
  ax25_address digipeaters[AX25_MAX_DIGIS];
};
#endif
