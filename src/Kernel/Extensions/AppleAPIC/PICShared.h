/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
#ifndef _IOKIT_PICSHARED_H
#define _IOKIT_PICSHARED_H

#if     APIC_DEBUG
#define APIC_LOG(args...)  kprintf(args)
#else
#define APIC_LOG(args...)
#endif

/*
 * First 32-bit value in the IOInterruptSpecifier data is
 * the vector number, followed by the interrupt flags.
 */
#define DATA_TO_VECTOR(data) \
        (((UInt32 *)(data)->getBytesNoCopy())[0])

#define DATA_TO_FLAGS(data)  \
        (((UInt32 *)(data)->getBytesNoCopy())[1])

/*
 * Interrupt flags returned by DATA_TO_FLAGS().
 */
enum {
    kInterruptTriggerModeMask  = 0x01,
    kInterruptTriggerModeEdge  = 0x00,
    kInterruptTriggerModeLevel = kInterruptTriggerModeMask,
    kInterruptPolarityMask     = 0x02,
    kInterruptPolarityHigh     = 0x00,
    kInterruptPolarityLow      = kInterruptPolarityMask,
    kInterruptShareableMask    = 0x04,
    kInterruptNotShareable     = 0x00,
    kInterruptIsShareable      = kInterruptShareableMask,
};

/*
 * Keys for properties in the interrupt controller device/nub.
 */
#define kInterruptControllerNameKey   "InterruptControllerName"
#define kDestinationAPICIDKey         "Destination APIC ID"
#define kBaseVectorNumberKey          "Base Vector Number"
#define kVectorCountKey               "Vector Count"
#define kPhysicalAddressKey           "Physical Address"
#define kTimerVectorNumberKey         "Timer Vector Number"

/*
 * callPlatformFunction function names.
 */
#define kHandleSleepWakeFunction      "HandleSleepWake"
#define kSetVectorPhysicalDestination "SetVectorPhysicalDestination"

#endif /* !_IOKIT_PICSHARED_H */
