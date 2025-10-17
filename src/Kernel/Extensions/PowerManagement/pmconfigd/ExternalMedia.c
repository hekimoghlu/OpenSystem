/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include <CoreFoundation/CoreFoundation.h>
#include <DiskArbitration/DiskArbitration.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <IOKit/storage/IOStorageProtocolCharacteristics.h>

#include "ExternalMedia.h"
#include "PMAssertions.h"
#include "PrivateLib.h"


#define _kExternalMediaAssertionName 	"com.apple.powermanagement.externalmediamounted"

static dispatch_queue_t  extMediaQ;

/*****************************************************************************/


static bool weLikeTheDisk(DADiskRef disk);
static void adjustExternalDiskAssertion(void);
static void _DiskDisappeared(DADiskRef disk, void *context);
static void _DiskAppeared(DADiskRef disk, void * context);

/*****************************************************************************/

static CFMutableSetRef      gExternalMediaSet = NULL;
static IOPMAssertionID      gDiskAssertionID = kIOPMNullAssertionID;
static DASessionRef         gDASession = NULL;

/*****************************************************************************/


__private_extern__ void ExternalMedia_prime(void)
{    
    extMediaQ = dispatch_queue_create("com.apple.powermanagent.extMediaQ", DISPATCH_QUEUE_SERIAL);

    dispatch_async(extMediaQ, ^{
        gExternalMediaSet = CFSetCreateMutable(0, 0, &kCFTypeSetCallBacks);
        
        if (!gExternalMediaSet)
            return;
        
        gDASession = DASessionCreate(0);
        
        DARegisterDiskAppearedCallback(gDASession, kDADiskDescriptionMatchVolumeMountable, _DiskAppeared, NULL);
        
        DARegisterDiskDisappearedCallback(gDASession, kDADiskDescriptionMatchVolumeMountable, _DiskDisappeared, NULL);
        
        DASessionSetDispatchQueue(gDASession, extMediaQ);
    });
    
}

/*****************************************************************************/

static void _DiskDisappeared(DADiskRef disk, void *context)
{
    if (weLikeTheDisk(disk))
    {		
        CFSetRemoveValue(gExternalMediaSet, disk);
        
        adjustExternalDiskAssertion();
    }
}

/*****************************************************************************/

static void _DiskAppeared(DADiskRef disk, void * context)
{
    if (weLikeTheDisk(disk))
    {		
        CFSetSetValue(gExternalMediaSet, disk);
        
        adjustExternalDiskAssertion();
    }
}


/*****************************************************************************/

static bool weLikeTheDisk(DADiskRef disk)
{
    CFDictionaryRef     description = NULL;
    CFStringRef         protocol = NULL;
    bool                ret = false;

    /*
    We will create an ExternalMedia assertion if any of these disks are present.
    That will prevent deep sleep.     
      USB hard drive    : Protocol = USB
      USB thumb drive   : Protocol = USB 
      SD Card           : Protocol = USB, Protocol = Secure Digital 
      External drive    : Interconnect Location = External

    These disks do not cause us to create an ExternalMedia assertion;
      CD/DVD            : Protocol = ATAPI
      Disk Image        : Protocol = Disk Image
    */
    
    description = DADiskCopyDescription(disk);
    if (description) {
        
        if (CFDictionaryGetValue(description, kDADiskDescriptionDeviceInternalKey) == kCFBooleanFalse) {
            ret = true;
        } else {
            protocol = CFDictionaryGetValue(description, kDADiskDescriptionDeviceProtocolKey);

            if (protocol &&
                (CFEqual(protocol, CFSTR(kIOPropertyPhysicalInterconnectTypeUSB)) ||
                 CFEqual(protocol, CFSTR(kIOPropertyPhysicalInterconnectTypeSecureDigital))))
            {
                ret = true;
            }
        }

        CFRelease(description);
    }
    return ret;
}

/*****************************************************************************/


static void adjustExternalDiskAssertion(void)
{
    CFIndex	deviceCount = CFSetGetCount(gExternalMediaSet);
    
    if (0 == deviceCount)
    {	
        /*
         * Release assertion
         */
        
        // This call dispatches assertion release to main queue
        InternalReleaseAssertion(&gDiskAssertionID);
        

        return;
    }
    

    if (0 < deviceCount)
    {
        /*
         *  Create new assertion
         */
         
        CFMutableDictionaryRef assertionDescription = NULL;
         
        assertionDescription = _IOPMAssertionDescriptionCreate(
                                        _kIOPMAssertionTypeExternalMedia, 
                                        CFSTR(_kExternalMediaAssertionName), 
                                        NULL, CFSTR("An external media device is attached."), NULL, 0, NULL);
        
        // This call dispatches assertion create to main queue
        InternalCreateAssertion(assertionDescription, 
                                &gDiskAssertionID);

        CFRelease(assertionDescription);

        return;
    }

    return;
}


