/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#ifndef __LP64__
#pragma options align=mac68k
#endif

enum
{
    kDVIPowerSwitchPowerOffDelay  = 200         /* ms before power off */
};

enum
{
    kIODVIPowerEnableFlag  = 0x00010000,
    kIOI2CPowerEnableFlag  = 0x00020000,
    kIONoncoherentTMDSFlag = 0x00040000
};

#define kIONDRVDisplayConnectFlagsKey   "display-connect-flags"

enum { kIONDRVAVJackProbeDelayMS = 1000 };

enum {
    cscSleepWake = 0x86,
    sleepWakeSig = 'slwk',
    vdSleepState = 0,
    vdWakeState  = 1
};

struct VDSleepWakeInfo
{
    UInt8       csMode;
    UInt8       fill;
    UInt32      csData;
};
typedef struct VDSleepWakeInfo VDSleepWakeInfo;


#ifndef __LP64__
#pragma options align=reset
#endif

