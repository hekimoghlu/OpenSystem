/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#include "open.h"

#include <errno.h>
#include <util.h>
#include <DiskArbitration/DiskArbitration.h>
#include <IOKit/storage/IOMedia.h>
#include <IOKit/IOKitLibPrivate.h>

static DASessionRef __GetSession( void )
{
    static dispatch_once_t once;
    static DASessionRef session;

    dispatch_once( &once, ^
    {
        session = DASessionCreate( kCFAllocatorDefault );

        if ( session )
        {
            DASessionSetDispatchQueue( session, dispatch_get_main_queue( ) );
        }
    } );

    return session;
}

static void __UnmountCallback( DADiskRef disk __unused, DADissenterRef dissenter __unused, void * context )
{
    dispatch_semaphore_t semaphore = ( __bridge id ) context;

    dispatch_semaphore_signal( semaphore );
}

static void __Unmount( DADiskRef disk )
{
    dispatch_semaphore_t semaphore;

    semaphore = dispatch_semaphore_create( 0 );

    if ( semaphore )
    {
        DADiskUnmount( disk, kDADiskUnmountOptionDefault, __UnmountCallback, ( __bridge void * ) semaphore );

        dispatch_semaphore_wait( semaphore, DISPATCH_TIME_FOREVER );
    }
}

int _Open( io_service_t service, int oflag, uint64_t authorizationID )
{
    uint64_t _authorizationID = 0;
    int fd = -1;
    IOReturn status;

    _IOServiceGetAuthorizationID( service, &_authorizationID );

    if ( _authorizationID )
    {
        if ( _authorizationID == authorizationID )
        {
            if ( IOObjectConformsTo( service, kIOMediaClass ) )
            {
                DADiskRef disk;

                disk = DADiskCreateFromIOMedia( kCFAllocatorDefault, __GetSession( ), service );

                if ( disk )
                {
                    CFDictionaryRef description;

                    description = DADiskCopyDescription( disk );

                    if ( description )
                    {
                        if ( CFDictionaryGetValue( description, kDADiskDescriptionDeviceInternalKey ) == kCFBooleanTrue )
                        {
                            if ( CFDictionaryGetValue( description, kDADiskDescriptionMediaRemovableKey ) == kCFBooleanFalse )
                            {
                                if ( CFDictionaryGetValue( description, kDADiskDescriptionVolumePathKey ) )
                                {
                                    __Unmount( disk );
                                }
                            }
                        }

                        CFRelease( description );
                    }

                    description = DADiskCopyDescription( disk );

                    if ( description )
                    {
                        if ( CFDictionaryGetValue( description, kDADiskDescriptionVolumePathKey ) == 0 )
                        {
                            fd = opendev( ( void * ) DADiskGetBSDName( disk ), oflag, 0, 0 );

                            if ( fd == -1 )
                            {
                                status = unix_err( errno );
                            }
                            else
                            {
                                status = kIOReturnSuccess;
                            }
                        }
                        else
                        {
                            status = unix_err( EBUSY );
                        }

                        CFRelease( description );
                    }
                    else
                    {
                        status = unix_err( EBUSY );
                    }

                    CFRelease( disk );
                }
                else
                {
                    status = unix_err( ENOTSUP );
                }
            }
            else
            {
                status = unix_err( ENOTSUP );
            }
        }
        else
        {
            status = unix_err( EPERM );
        }
    }
    else
    {
        status = unix_err( EPERM );
    }

    if ( status )
    {
        if ( unix_err( err_get_code( status ) ) == status )
        {
            errno = err_get_code( status );
        }
    }

    return fd;
}
