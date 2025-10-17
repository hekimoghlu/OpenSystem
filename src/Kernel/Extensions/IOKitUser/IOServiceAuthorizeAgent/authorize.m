/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "authorize.h"

#include "application.h"
#include "device.h"
#include "preferences.h"

#include <CoreFoundation/CFBundlePriv.h>
#include <IOKit/IOKitLibPrivate.h>

static IOReturn __Authorize( CFDictionaryRef deviceID, CFBundleRef bundle )
{
    CFStringRef device;
    IOReturn status;

    device = _DeviceCopyName( deviceID );

    if ( device )
    {

        CFStringRef application;

        application = CFBundleGetValueForInfoDictionaryKey( bundle, _kCFBundleDisplayNameKey );

        if ( application == 0 )
        {
            application = CFBundleGetValueForInfoDictionaryKey( bundle, kCFBundleNameKey );
        }

        if ( application )
        {
            CFMutableDictionaryRef dictionary;

            dictionary = CFDictionaryCreateMutable( kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );

            if ( dictionary )
            {
                CFStringRef string;
                CFStringRef header;

                string = CFCopyLocalizedString( CFSTR( "%@ wants to access \"%@\"." ), 0 );
                header = CFStringCreateWithFormat( kCFAllocatorDefault, 0, string, application, device );
                CFRelease( string );

                if ( header )
                {
                    CFURLRef path;

                    path = CFBundleCopyBundleURL( CFBundleGetMainBundle( ) );

                    if ( path )
                    {
                        CFUserNotificationRef notification;

                        CFDictionarySetValue( dictionary, kCFUserNotificationAlertHeaderKey, header );
                        CFDictionarySetValue( dictionary, kCFUserNotificationAlertMessageKey, CFSTR( "Do you want to allow access to this device?" ) );
                        CFDictionarySetValue( dictionary, kCFUserNotificationAlternateButtonTitleKey, CFSTR( "Always Allow" ) );
                        CFDictionarySetValue( dictionary, kCFUserNotificationDefaultButtonTitleKey, CFSTR( "Allow" ) );
                        CFDictionarySetValue( dictionary, kCFUserNotificationLocalizationURLKey, path );
                        CFDictionarySetValue( dictionary, kCFUserNotificationOtherButtonTitleKey, CFSTR( "Deny" ) );

                        notification = CFUserNotificationCreate( kCFAllocatorDefault, 0, kCFUserNotificationCautionAlertLevel, 0, dictionary );

                        if ( notification )
                        {
                            int err;
                            CFOptionFlags response;

                            err = CFUserNotificationReceiveResponse( notification, 0, &response );

                            if ( err == 0 )
                            {
                                switch ( ( response & 0x3 ) )
                                {
                                    case kCFUserNotificationAlternateResponse:
                                    {
                                        status = kIOReturnNotFound;

                                        break;
                                    }

                                    case kCFUserNotificationDefaultResponse:
                                    {
                                        status = kIOReturnSuccess;

                                        break;
                                    }

                                    default:
                                    {
                                        status = kIOReturnNotPermitted;

                                        break;
                                    }
                                }
                            }
                            else
                            {
                                status = kIOReturnNoResources;
                            }

                            CFRelease( notification );
                        }
                        else
                        {
                            status = kIOReturnNoResources;
                        }

                        CFRelease( path );
                    }
                    else
                    {
                        status = kIOReturnNoMemory;
                    }

                    CFRelease( header );
                }
                else
                {
                    status = kIOReturnNoMemory;
                }

                CFRelease( dictionary );
            }
            else
            {
                status = kIOReturnNoMemory;
            }
        }
        else
        {
            status = kIOReturnAborted;
        }

        CFRelease( device );
    }
    else
    {
        status = kIOReturnUnsupported;
    }

    return status;
}

IOReturn _Authorize( io_service_t service, uint64_t options, pid_t processID, uint64_t authorizationID, const audit_token_t *auditToken )
{
    CFDictionaryRef deviceID;
    IOReturn status;

    deviceID = _DeviceCopyIdentifier( service );

    if ( deviceID )
    {

        CFBundleRef bundle;

        bundle = _ApplicationCopyBundle( processID );

        if (bundle)
        {
            CFStringRef applicationID;

            applicationID = _ApplicationCopyIdentifier( processID, auditToken );

            if ( applicationID )
            {
                CFArrayRef deviceIDs;

                status = kIOReturnNotFound;

                deviceIDs = _PreferencesCopyValue( applicationID );

                if ( deviceIDs )
                {
                    CFIndex count;
                    CFIndex index;

                    count = CFArrayGetCount( deviceIDs );

                    for ( index = 0; index < count; index++ )
                    {
                        CFDictionaryRef compare;

                        compare = ( void * ) CFArrayGetValueAtIndex( deviceIDs, index );

                        if ( _DeviceIsEqual( deviceID, compare ) )
                        {
                            status = kIOReturnSuccess;

                            break;
                        }
                    }
                }

                if ( status )
                {
                    if ( ( options & kIOServiceInteractionAllowed ) )
                    {
                        status = __Authorize( deviceID, bundle );

                        if ( status == kIOReturnNotFound )
                        {
                            _PreferencesAppendArrayValue( applicationID, deviceID );

                            status = kIOReturnSuccess;
                        }
                    }
                    else
                    {
                        status = kIOReturnNotPermitted;
                    }
                }

                if ( status == kIOReturnSuccess )
                {
                    status = _IOServiceSetAuthorizationID( service, authorizationID );
                }

                if ( deviceIDs )
                {
                    CFRelease( deviceIDs );
                }

                CFRelease( applicationID );
            }
            else
            {
                status = kIOReturnAborted;
            }

            CFRelease( bundle );
        }
        else
        {
            status = kIOReturnAborted;
        }

        CFRelease( deviceID );
    }
    else
    {
        status = kIOReturnUnsupported;
    }

    return status;
}
