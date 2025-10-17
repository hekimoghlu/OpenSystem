/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/vm_map.h>
#include <servers/bootstrap.h>


#include "IOSystemConfiguration.h"
#include <CoreFoundation/CoreFoundation.h> 
#include <IOKit/IOKitLib.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <IOKit/pwr_mgt/IOPMLibPrivate.h>
#include "powermanagement_mig.h"
#include "powermanagement.h"

__private_extern__ IOReturn _copyPMServerObject(int selector, int assertionID,
                                                CFTypeRef selectorData, CFTypeRef *objectOut);

/*
 * SCPreferences file format
 *     com.apple.AutoWake.xml
 *
 * - CFSTR(kIOPMRepeatingPowerOnKey)
 *      - CFSTR(kIOPMPowerEventTimeKey) = CFNumberRef (kCFNumberIntType)
 *      - CFSTR(kIOPMDaysOfWeekKey) = CFNumberRef (kCFNumberIntType)
 *      - CFSTR(kIOPMPowerEventTypeKey) = CFStringRef (kIOPMAutoSleep, kIOPMAutoShutdown, kIOPMAutoPowerOn, kIOPMAutoWake)
 * - CFSTR(kIOPMRepeatingPowerOffKey)
 *      - CFSTR(kIOPMPowerEventTimeKey) = CFNumberRef (kCFNumberIntType)
 *      - CFSTR(kIOPMDaysOfWeekKey) = CFNumberRef (kCFNumberIntType)
 *      - CFSTR(kIOPMPowerEventTypeKey) = CFStringRef (kIOPMAutoSleep, kIOPMAutoShutdown, kIOPMAutoPowerOn, kIOPMAutoWake)
 */
 
IOReturn _pm_connect(mach_port_t *newConnection);
IOReturn _pm_disconnect(mach_port_t connection);

IOReturn IOPMScheduleRepeatingPowerEvent(CFDictionaryRef events)
{
    IOReturn                    ret = kIOReturnError;
    CFDataRef                   flatPackage = NULL;
    kern_return_t               rc = KERN_SUCCESS;
    mach_port_t                 pm_server = MACH_PORT_NULL;
    
    // Validate our inputs
    if(!isA_CFDictionary(events)) return kIOReturnBadArgument;

    
    if(kIOReturnSuccess != _pm_connect(&pm_server)) {
        ret = kIOReturnInternalError;
        goto exit;
    }

    flatPackage = CFPropertyListCreateData(0, events,
                          kCFPropertyListBinaryFormat_v1_0, 0, NULL );

    if ( !flatPackage ) {
        ret = kIOReturnBadArgument;
        goto exit;
    }

    rc = io_pm_schedule_repeat_event(pm_server, (vm_offset_t)CFDataGetBytePtr(flatPackage),
            CFDataGetLength(flatPackage), 1, &ret);

    if (rc != KERN_SUCCESS)
        ret = kIOReturnInternalError;
 
exit:

    if (MACH_PORT_NULL != pm_server) {
        _pm_disconnect(pm_server);
    }
    if(flatPackage) CFRelease(flatPackage);

    return ret;
}


CFDictionaryRef IOPMCopyRepeatingPowerEvents(void)
{
    CFMutableDictionaryRef      return_dict = NULL;

    _copyPMServerObject(kIOPMPowerEventsMIGCopyRepeatEvents, 0, NULL, (CFTypeRef *)&return_dict);
    return return_dict;
}

IOReturn IOPMCancelAllRepeatingPowerEvents(void)
{    
    IOReturn                    ret = kIOReturnError;
    kern_return_t               rc = KERN_SUCCESS;
    mach_port_t                 pm_server = MACH_PORT_NULL;
    
    if(kIOReturnSuccess != _pm_connect(&pm_server)) {
        ret = kIOReturnInternalError;
        goto exit;
    }

    rc = io_pm_cancel_repeat_events(pm_server, &ret);

    if (rc != KERN_SUCCESS)
        ret = kIOReturnInternalError;


    if (MACH_PORT_NULL != pm_server) {
        _pm_disconnect(pm_server);
    }

exit:
    return ret;
}
