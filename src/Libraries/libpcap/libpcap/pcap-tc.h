/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#ifndef __PCAP_TC_H__
#define __PCAP_TC_H__

/*
 * needed because gcc headers do not have C_ASSERT
 */
#ifndef C_ASSERT
#define C_ASSERT(a)
#endif

#include <TcApi.h>

/*
 * functions used effectively by the pcap library
 */

pcap_t *
TcCreate(const char *device, char *ebuf, int *is_ours);

int
TcFindAllDevs(pcap_if_list_t *devlistp, char *errbuf);

#endif
