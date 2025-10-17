/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
#ifndef _NET_IF_UTUN_H_
#define _NET_IF_UTUN_H_

#include <sys/types.h>

#ifdef KERNEL_PRIVATE

#include <sys/kern_control.h>

void* utun_alloc(size_t size);
void utun_free(void *ptr);
errno_t utun_register_control(void);
#if SKYWALK
boolean_t utun_interface_needs_netagent(ifnet_t interface);
#endif /* SKYWALK */

#endif

/*
 * Name registered by the utun kernel control
 */
#define UTUN_CONTROL_NAME "com.apple.net.utun_control"

/*
 * Socket option names to manage utun
 */
#define UTUN_OPT_FLAGS                                  1
#define UTUN_OPT_IFNAME                                 2
#define UTUN_OPT_EXT_IFDATA_STATS                       3       /* get|set (type int) */
#define UTUN_OPT_INC_IFDATA_STATS_IN                    4       /* set to increment stat counters (type struct utun_stats_param) */
#define UTUN_OPT_INC_IFDATA_STATS_OUT                   5       /* set to increment stat counters (type struct utun_stats_param) */

#define UTUN_OPT_SET_DELEGATE_INTERFACE                 15      /* set the delegate interface (char[]) */
#define UTUN_OPT_MAX_PENDING_PACKETS                    16      /* the number of packets that can be waiting to be read
	                                                         * from the control socket at a time */
#define UTUN_OPT_ENABLE_CHANNEL                         17
#define UTUN_OPT_GET_CHANNEL_UUID                       18
#define UTUN_OPT_ENABLE_FLOWSWITCH                      19

#define UTUN_OPT_ENABLE_NETIF                           20      /* Must be set before connecting */
#define UTUN_OPT_SLOT_SIZE                              21      /* Must be set before connecting */
#define UTUN_OPT_NETIF_RING_SIZE                        22      /* Must be set before connecting */
#define UTUN_OPT_TX_FSW_RING_SIZE                       23      /* Must be set before connecting */
#define UTUN_OPT_RX_FSW_RING_SIZE                       24      /* Must be set before connecting */
#define UTUN_OPT_KPIPE_TX_RING_SIZE                     25      /* Must be set before connecting */
#define UTUN_OPT_KPIPE_RX_RING_SIZE                     26      /* Must be set before connecting */
#define UTUN_OPT_ATTACH_FLOWSWITCH                      27      /* Must be set before connecting */

/*
 * Flags for by UTUN_OPT_FLAGS
 */
#define UTUN_FLAGS_NO_OUTPUT            0x0001
#define UTUN_FLAGS_NO_INPUT             0x0002
#define UTUN_FLAGS_ENABLE_PROC_UUID     0x0004

/*
 * utun stats parameter structure
 */
struct utun_stats_param {
	u_int64_t       utsp_packets;
	u_int64_t       utsp_bytes;
	u_int64_t       utsp_errors;
};

#endif
