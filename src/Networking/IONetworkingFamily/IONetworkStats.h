/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#ifndef _IONETWORKSTATS_H
#define _IONETWORKSTATS_H

#include <libkern/OSTypes.h>

/*! @header IONetworkStats.h
    @discussion Generic network statistics. */

//------------------------------------------------------------------------
// Generic network statistics. Common to all network interfaces.
//
// WARNING: This structure must match the statistics field in
// ifnet->if_data. This structure will overlay a portion of ifnet.

/*! @typedef IONetworkStats
        @discussion Generic network statistics structure.
        @field inputPackets count input packets.
        @field inputErrors count input errors.
        @field outputPackets count output packets.
        @field outputErrors count output errors.
        @field collisions count collisions on CDMA networks. */

typedef struct {
        UInt32  inputPackets;
        UInt32  inputErrors;
        UInt32  outputPackets;
        UInt32  outputErrors;
        UInt32  collisions;
} IONetworkStats;

/*! @defined kIONetworkStatsKey
        @discussion Defines the name of an IONetworkData that contains
        an IONetworkStats. */

#define kIONetworkStatsKey              "IONetworkStatsKey"

//------------------------------------------------------------------------
// Output queue statistics.

/*! @typedef IOOutputQueueStats
        @discussion Statistics recorded by IOOutputQueue objects.
        @field capacity queue capacity.
        @field size current size of the queue.
        @field peakSize peak size of the queue.
        @field dropCount number of packets dropped.
        @field outputCount number of output packets.
        @field retryCount number of retries.
        @field stallCount number of queue stalls. */

typedef struct {
        UInt32  capacity;
        UInt32  size;
        UInt32  peakSize;
        UInt32  dropCount;
        UInt32  outputCount;
        UInt32  retryCount;
        UInt32  stallCount;
        UInt32  reserved[4];
} IOOutputQueueStats;

/*! @defined kIOOutputQueueStatsKey
        @discussion Defines the name of an IONetworkData that contains
        an IOOutputQueueStats. */

#define kIOOutputQueueStatsKey          "IOOutputQueueStatsKey"

#endif /* !_IONETWORKSTATS_H */
