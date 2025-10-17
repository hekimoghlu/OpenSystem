/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef _IONETWORKCONTROLLERPRIVATE_H
#define _IONETWORKCONTROLLERPRIVATE_H

#define kMessageControllerWasEnabled  \
        iokit_family_msg(sub_iokit_networking, 0x110)

#define kMessageControllerWasDisabled \
        iokit_family_msg(sub_iokit_networking, 0x111)

#define kMessageControllerWasEnabledForBSD  \
        iokit_family_msg(sub_iokit_networking, 0x112)

#define kMessageControllerWasDisabledForBSD \
        iokit_family_msg(sub_iokit_networking, 0x113)

#define kMessageControllerWillShutdown \
        iokit_family_msg(sub_iokit_networking, 0x114)

#define kMessageDebuggerActivationChange \
        iokit_family_msg(sub_iokit_networking, 0x1F0)

// kIONetworkEventTypeLink message payload
struct IONetworkLinkEventData {
    uint64_t    linkSpeed;
    uint32_t    linkStatus;
    uint32_t    linkType;
};

#define kGPTPPresentKey "gPTPPresent"
#define kTimeSyncSupportKey "TimeSyncSupport"
#define kAVBControllerStateKey "AVBControllerState"
#define kNumberOfRealtimeTransmitQueuesKey "NumberOfRealtimeTransmitQueues"
#define kNumberOfRealtimeReceiveQueuesKey "NumberOfRealtimeReceiveQueues"

#ifdef __x86_64__
#define kIONetworkMbufDMADriversKey         "IONetworkMbufDMADrivers"

#ifdef KERNEL
extern OSArray * gIONetworkMbufCursorKexts;
extern IOLock *  gIONetworkMbufCursorLock;
extern IORegistryEntry * gIONetworkMbufCursorEntry;
#endif
#endif

#endif /* !_IONETWORKCONTROLLERPRIVATE_H */
