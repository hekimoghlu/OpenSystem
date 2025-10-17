/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#include "device.h"

#include "storage/storage.h"
#include "usb/usb.h"

#include <IOKit/storage/IOMedia.h>
#include <IOKit/usb/IOUSBLib.h>

CFDictionaryRef _DeviceCopyIdentifier( io_service_t service )
{
    CFDictionaryRef identifier = 0;

    if ( IOObjectConformsTo( service, kIOMediaClass ) )
    {
        identifier = _IOMediaCopyIdentifier( service );
    }
    else if ( IOObjectConformsTo( service, kIOUSBDeviceClassName ) )
    {
        identifier = _IOUSBDeviceCopyIdentifier( service );
    }
    else if ( IOObjectConformsTo( service, "IOUSBDevice" ) )
    {

        identifier = _IOUSBDeviceCopyIdentifier( service );
    }
    return identifier;
}

CFStringRef _DeviceCopyName( CFDictionaryRef identifier )
{
    CFStringRef class;
    CFStringRef name = 0;

    class = CFDictionaryGetValue( identifier, CFSTR( kIOProviderClassKey ) );

    if ( CFEqual( class, CFSTR( kIOMediaClass ) ) )
    {
        name = _IOMediaCopyName( identifier );
    }
    else if ( CFEqual( class, CFSTR( kIOUSBDeviceClassName ) ) )
    {
        name = _IOUSBDeviceCopyName( identifier );
    }
    else if ( CFEqual( class, CFSTR( "IOUSBDevice" ) ) )
    {
        name = _IOUSBDeviceCopyName( identifier );
    }

    return name;
}

Boolean _DeviceIsEqual( CFDictionaryRef identifier1, CFDictionaryRef identifier2 )
{
    CFStringRef class;
    Boolean equal = FALSE;

    class = CFDictionaryGetValue( identifier1, CFSTR( kIOProviderClassKey ) );

    if ( CFEqual( class, CFDictionaryGetValue( identifier2, CFSTR( kIOProviderClassKey ) ) ) )
    {
        if ( CFEqual( class, CFSTR( kIOMediaClass ) ) )
        {
            equal = _IOMediaIsEqual( identifier1, identifier2 );
        }
        else if ( CFEqual( class, CFSTR( kIOUSBDeviceClassName ) ) )
        {
            equal = _IOUSBDeviceIsEqual( identifier1, identifier2 );
        }
        else if ( CFEqual( class, CFSTR( "IOUSBDevice" ) ) )
        {
            equal = _IOUSBDeviceIsEqual( identifier1, identifier2 );
        }
    }

    return equal;
}

Boolean _DeviceIsValid( io_service_t service )
{
    Boolean valid = FALSE;

    if ( IOObjectConformsTo( service, kIOMediaClass ) )
    {
        valid = _IOMediaIsValid( service );
    }
    else if ( IOObjectConformsTo( service, kIOUSBDeviceClassName ) )
    {
        valid = _IOUSBDeviceIsValid( service );
    }
    else if ( IOObjectConformsTo( service, "IOUSBDevice" ) )
    {
        valid = _IOUSBDeviceIsValid( service );
    }

    return valid;
}
