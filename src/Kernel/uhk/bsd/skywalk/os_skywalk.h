/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#ifndef _SKYWALK_OS_SKYWALK_H
#define _SKYWALK_OS_SKYWALK_H

#ifdef PRIVATE
#include <skywalk/os_channel.h>
#include <skywalk/os_nexus.h>
#include <skywalk/os_packet.h>
#endif /* PRIVATE */

/*
 * Skywalk ktrace event ID
 *
 * Always on events are captured by artrace by default. Others can be
 * selectively enabled via artrace -f S0x08[subclass], whereas [subclass] is
 * one of DBG_SKYWALK_{ALWAYSON, FLOWSWITCH, NETIF, CHANNEL, PACKET}.
 *
 * Please keep values in sync with skywalk_signposts.plist and assertions in
 * skywalk_self_tests.
 */
/** @always-on subclass */
#define SK_KTRACE_AON_IF_STATS                  SKYWALKDBG_CODE(DBG_SKYWALK_ALWAYSON, 0x001)

/** @flowswitch subclass */
#define SK_KTRACE_FSW_DEV_RING_FLUSH            SKYWALKDBG_CODE(DBG_SKYWALK_FLOWSWITCH, 0x001)
#define SK_KTRACE_FSW_USER_RING_FLUSH           SKYWALKDBG_CODE(DBG_SKYWALK_FLOWSWITCH, 0x002)
#define SK_KTRACE_FSW_FLOW_TRACK_RTT            SKYWALKDBG_CODE(DBG_SKYWALK_FLOWSWITCH, 0x004)

/** @netif subclass */
#define SK_KTRACE_NETIF_RING_TX_REFILL          SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x001)
#define SK_KTRACE_NETIF_HOST_ENQUEUE            SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x002)
#define SK_KTRACE_NETIF_MIT_RX_INTR             SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x003)
#define SK_KTRACE_NETIF_COMMON_INTR             SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x004)
#define SK_KTRACE_NETIF_RX_NOTIFY_DEFAULT       SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x005)
#define SK_KTRACE_NETIF_RX_NOTIFY_FAST          SKYWALKDBG_CODE(DBG_SKYWALK_NETIF, 0x006)

/** @channel subclass */
#define SK_KTRACE_CHANNEL_TX_REFILL             SKYWALKDBG_CODE(DBG_SKYWALK_CHANNEL, 0x1)

/** @packet subclass */
/*
 * Used with os_packet_trace_* functions.
 * Total of 12bit (0xABC) code space available, current sub-code allocation is:
 *     0x00C code space for FSW Rx path.
 *     0x01C code space for FSW Tx path.
 * More sub-code can be added for other packet data path, e.g. uPipe, BSD, etc.
 */
/* @packet::rx group */
#define SK_KTRACE_PKT_RX_DRV                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x001)
#define SK_KTRACE_PKT_RX_FSW                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x002)
#define SK_KTRACE_PKT_RX_CHN                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x003)
/* @packet::tx group */
#define SK_KTRACE_PKT_TX_FSW                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x010)
#define SK_KTRACE_PKT_TX_AQM                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x011)
#define SK_KTRACE_PKT_TX_DRV                    SKYWALKDBG_CODE(DBG_SKYWALK_PACKET, 0x012)

#endif /* _SKYWALK_OS_SKYWALK_H */
