/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
/*
 * Copyright (c) 2002 Apple Computer, Inc.  All rights reserved. 
 *
 * HISTORY
 *
 * 29-Aug-02 ebold created
 *
 */
 
#ifndef _PMSettings_h_
#define _PMSettings_h_

#include "PrivateLib.h"

/* Power Management profile bits */
enum {
    kPMForceLowSpeedProfile         = (1<<0),
    kPMForceHighSpeed               = (1<<1),
    kPMPreventIdleSleep             = (1<<2),
    kPMPreventDisplaySleep          = (1<<3),
    kPMPreventDiskSleep             = (1<<4),
    kPMPreventWakeOnLan             = (1<<5)
};

__private_extern__ void PMSettings_prime(void);
 
__private_extern__ void PMSettingsCapabilityChangeNotification(const struct IOPMSystemCapabilityChangeParameters * p);

__private_extern__ void PMSettingsSupportedPrefsListHasChanged(void);

__private_extern__ void PMSettingsPrefsHaveChanged(void);

__private_extern__ void PMSettingsPSChange(void);

__private_extern__ bool GetSystemPowerSettingBool(CFStringRef);

__private_extern__ bool GetPMSettingBool(CFStringRef);

__private_extern__ IOReturn GetPMSettingNumber(CFStringRef which, int64_t *value);

// For UPS shutdown/restart code in PSLowPower.c
__private_extern__ CFDictionaryRef  PMSettings_CopyActivePMSettings(void);

__private_extern__ IOReturn _activateForcedSettings(CFDictionaryRef);

// For IOPMAssertions code in PMAssertions.c
__private_extern__ void overrideSetting(int, int);
__private_extern__ void activateSettingOverrides(void);

__private_extern__ IOReturn getDisplaySleepTimer(uint32_t *displaySleepTimer);
__private_extern__ IOReturn getIdleSleepTimer(unsigned long *idleSleepTimer);
__private_extern__ void setDisplaySleepFactor(unsigned int factor);
__private_extern__ void setDisplayToDimTimer(io_connect_t connection, unsigned int minutesToDim);

__private_extern__ void saveAlarmInfo(CFDictionaryRef info);
__private_extern__ CFDictionaryRef copyAlarmInfo(void);

__private_extern__ bool _DWBT_allowed(void);
__private_extern__ bool _DWBT_enabled(void);

__private_extern__ bool _SS_allowed(void);

#ifdef XCTEST
void xctSetPowerNapState(bool allowDBT, bool allowSS);
void xctSetEnergySettings(CFDictionaryRef settings);
#endif


#endif //_PMSettings_h_
