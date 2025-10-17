/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#ifndef _IOPMUPSPrivate_h_
#define _IOPMUPSPrivate_h_

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOReturn.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Keys for the CFDictionary used to communicate with the UPS preferences.
 * Caller must wrap these with CFStrings.
 *
 * kIOUPSShutdownAtLevelKey, kIOUPSShutdownAfterMinutesOn, kIOUPSShutdownAtMinutesLeft
 *
 * Each key corresponds to a CFDictionary that contains:
 *      kIOUPSShutdownLevelEnabledKey - A CFBooleanRef for "enabled" or not
 *      kIOUPSShutdownLevelValueKey - A CFNumber, of type kCFNumberIntType, containing the % or minute setting
 *
 * This is the dictionary format used by IOPMSetUPSPreferences & IOPMCopyUPSPreferences.
 *
 */
#define     kIOUPSShutdownLevelEnabledKey   "Enabled"
#define     kIOUPSShutdownLevelValueKey     "Value"


// Shutdown level value is a CFNumber, kCFNumberIntType, 0-100%
#define     kIOUPSShutdownAtLevelKey        "UPSShutdownAtLevel"

// Shutdown after minutes on is a CFNumber, kCFNumberIntType, >= 0
#define     kIOUPSShutdownAfterMinutesOn    "UPSShutdownAfterMinutes"

// Shutdown at minutes left on UPS is a CFNumber, kCFNumberIntType, >0
#define     kIOUPSShutdownAtMinutesLeft     "UPSShutdownAtMinutesLeft"


/* Value for whichUPS
 *
 * As of Rohan, we haven't determined how to uniquely identify UPS's.
 * For now, the only argument you should pass to whichUPS is
 * CFSTR(kIOUPSDefaultUPSSettings)
 *
 * The settings dictionary associtaed with kIOUPSDefaultUPSSettings
 *    will apply to all UPS's.
 *
 */
#define kIOPMDefaultUPSThresholds             "UPSDefaultThresholds"

    /*!
@function IOPMSetUPSShutdownPreferences.
@abstract Set shutdown settings for the given UPS.
@param whichUPS Identify the UPS you want to set shutdown thresholds for.
    For now, we don't have a unique way to identify UPS's. Settings will apply
    to all attached & active UPS's. Please pass in CFSTR(kIOPMDefaultUPSThresholds) here.
@param UPSPrefs A dictionary of settings as described above.
@result Returns kIOReturnSuccess or an error condition if request failed.
    kIOReturnBadArgument - Badly structured UPSPrefs dictionary.
    kIOReturnNotPrivileged - Caller does not have admin/root privileges to write preferences file.
    kIOReturnError - General error.
     */
IOReturn IOPMSetUPSShutdownLevels(CFTypeRef whichUPS, CFDictionaryRef UPSPrefs);

    /*!
@function IOPMCopyUPSPreferences.
@abstract Copy the current settings for the given UPS.
@param whichUPS Identify the UPS you want to set shutdown thresholds for.
    For now, we don't have a unique way to identify UPS's. Please pass in CFSTR(kIOPMDefaultUPSThresholds).
@result Returns a CFDictionary of settings with three key/value pairs in it.
    If a particular key is not present in the dictionary, then the given UPS
    does not support it.
    If NULL is returned, no matching UPS was found.
     */
CFDictionaryRef IOPMCopyUPSShutdownLevels(CFTypeRef whichUPS);

__END_DECLS

#endif
