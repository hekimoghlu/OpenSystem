/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#include "PrivateLib.h"
#include "PMSystemEvents.h"
#include "PMSettings.h"

#ifndef _PMSystemEvents_h_
#define _PMSystemEvents_h_

#include <notify.h>

#define THERMAL_NOTIFICATION_DIR  "/private/var/run/thermal"
#define THERMAL_NOTIFICATION_FILE "/private/var/run/thermal/.thermalpressure"

#define kIOPMRootDomainPowerStatusKey       "Power Status"

#define kMySCIdentity           CFSTR("IOKit Power")

static int              thermalState = kIOPMThermalLevelUnknown;
static int              perfState    = kIOPMPerformanceNormal;

static CFStringRef createSCKeyForIOKitString(CFStringRef str)
{
    CFStringRef     keyForString = NULL;

    if (CFEqual(str, CFSTR(kIOPMThermalLevelWarningKey))) 
    {
        keyForString = CFSTR("ThermalWarning");
    } else if (CFEqual(str, CFSTR(kIOPMCPUPowerLimitsKey))) {
        keyForString = CFSTR("CPUPower");
    } else if (CFEqual(str, CFSTR(kIOPMPerformanceWarningKey))) {
        keyForString = CFSTR("PerformanceWarning");
    }

    if (!keyForString)
        return NULL;

    return SCDynamicStoreKeyCreate(kCFAllocatorDefault, 
                        CFSTR("%@%@/%@"),
                        kSCDynamicStoreDomainState, 
                        CFSTR("/IOKit/Power"),
                        keyForString);
}

static const char * getNotifyKeyForIOKitString(CFStringRef str)
{
    if (CFEqual(str, CFSTR(kIOPMThermalLevelWarningKey))) 
    {
        return kIOPMThermalWarningNotificationKey;
    } else if (CFEqual(str, CFSTR(kIOPMCPUPowerLimitsKey))) 
    {
        return kIOPMCPUPowerNotificationKey;
    } else if (CFEqual(str, CFSTR(kIOPMPerformanceWarningKey))) 
    {
        return kIOPMPerformanceWarningNotificationKey;
    }
    return NULL;
}

    
__private_extern__ void 
PMSystemEvents_prime(void)
{
    // Publish default settings
    PMSystemEventsRootDomainInterest();    
}

__private_extern__ void 
PMSystemEventsRootDomainInterest(void)
{
    CFDictionaryRef         thermalStatus;
    CFMutableDictionaryRef  setTheseDSKeys = NULL;
    CFStringRef             *keys = NULL;
    CFNumberRef             *vals = NULL;
    SCDynamicStoreRef       store = NULL;
    CFIndex                 count = 0;
    CFIndex                 i;
    int                     thermNewState = -1;
    int                     perfNewState = -1;
    int                     create_file = 0;

    // Read dictionary from IORegistry
    thermalStatus = IORegistryEntryCreateCFProperty(
                            getRootDomain(),
                            CFSTR(kIOPMRootDomainPowerStatusKey),
                            kCFAllocatorDefault,
                            kNilOptions);

    if (!thermalStatus
        || !(count = CFDictionaryGetCount(thermalStatus)))
    {
        goto exit;
    }
    
    // Publish dictionary in SCDynamicStore
    keys = (CFStringRef *)malloc(count*sizeof(CFStringRef));
    vals = (CFNumberRef *)malloc(count*sizeof(CFNumberRef));
    if (!keys||!vals) 
        goto exit;
        
    CFDictionaryGetKeysAndValues(thermalStatus, 
                    (const void **)keys, (const void **)vals);
    
    setTheseDSKeys = CFDictionaryCreateMutable(0, count, 
                    &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (!setTheseDSKeys)
        goto exit;
    
    for (i=0; i<count; i++) 
    {
        CFStringRef writeToKey = createSCKeyForIOKitString(keys[i]);
        if (writeToKey) {
            CFDictionarySetValue(setTheseDSKeys, writeToKey, vals[i]);
            CFRelease(writeToKey);
        }
        if (CFStringCompare(keys[i], CFSTR(kIOPMThermalLevelWarningKey), 0) == kCFCompareEqualTo) {
            if (isA_CFNumber(vals[i])) {
                CFNumberGetValue(vals[i], kCFNumberIntType, &thermNewState);
                if (thermNewState != thermalState) {
                    int opVal = ((thermNewState == kIOPMThermalLevelWarning) || 
                                 (thermNewState == kIOPMThermalLevelTrap)) ? 1 : 0;
                    overrideSetting(kPMPreventWakeOnLan, opVal);
                    activateSettingOverrides();
                    if (thermNewState != kIOPMThermalLevelUnknown) {
                        logASLThermalState(thermNewState);
                    }
                }
            }
        }
        else if (CFStringCompare(keys[i], CFSTR(kIOPMPerformanceWarningKey), 0) == kCFCompareEqualTo) {
            if (isA_CFNumber(vals[i])) {
                CFNumberGetValue(vals[i], kCFNumberIntType, &perfNewState);
                if (perfNewState != perfState) {
                    logASLPerforamceState(perfNewState);
                }
            }
        }
    }

    store = SCDynamicStoreCreate(0, kMySCIdentity, NULL, NULL);
    if (!store)
        goto exit;

    SCDynamicStoreSetMultiple(store, setTheseDSKeys, NULL, NULL);

    for (i=0; i<count; i++)
    {
        if (CFStringCompare(keys[i], CFSTR(kIOPMThermalLevelWarningKey), 0) == kCFCompareEqualTo) {
            if (thermNewState != thermalState) {
                thermalState = thermNewState;

                if ((thermalState == kIOPMThermalLevelWarning) || (thermalState == kIOPMThermalLevelTrap)) {
                    create_file = 1;
                }
            }
            else {
                // Avoid notify_post call when thermal warning level hasn't changed
                continue;
            }
        }
        else if (CFStringCompare(keys[i], CFSTR(kIOPMPerformanceWarningKey), 0) == kCFCompareEqualTo) {
            if (perfNewState != perfState) {
                perfState = perfNewState;
                if (perfState == kIOPMPerformanceWarning) {
                    create_file = 1;
                }
            }
            else {
                // Avoid notify_post call when performance  warning level hasn't changed
                continue;
            }
        }

        const char *notify3Key = getNotifyKeyForIOKitString(keys[i]);
        if (notify3Key) 
            notify_post(notify3Key);
    }
    if (create_file) {
        int fd;
        
        if (!mkdir(THERMAL_NOTIFICATION_DIR, 0777)) {
            lchmod(THERMAL_NOTIFICATION_DIR, 0777);
        }

        fd = open(THERMAL_NOTIFICATION_FILE, O_CREAT|O_RDWR|O_NOFOLLOW_ANY, 0777);
        if (fd >= 0) {
            struct stat ss = {0};
            if (!fstat(fd, &ss) && (ss.st_nlink == 1)) {
                // Don't change the mode of hard linked files
                fchmod(fd, 0777);
            }
            close(fd);
        }
    }

exit:    
    if (keys)
        free(keys);
    if (vals)
        free(vals);
    if (setTheseDSKeys)
        CFRelease(setTheseDSKeys);
    if (store)
        CFRelease(store);
    if (thermalStatus)
        CFRelease(thermalStatus);
    return;
}


#endif //_PMSystemEvents_h_
