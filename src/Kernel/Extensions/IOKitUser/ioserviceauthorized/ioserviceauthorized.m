/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

#include <libproc.h>
#include <xpc/private.h>
#include <IOKit/IOKitLib.h>

static void __MessageCallback( xpc_connection_t connection, xpc_object_t message, uint64_t authorizationID )
{
    xpc_type_t type;

    type = xpc_get_type( message );

    if ( type == XPC_TYPE_DICTIONARY )
    {
        int64_t oflag;
        uint64_t serviceID;
        io_service_t service;
        int fd;
        IOReturn status;
        xpc_object_t reply;

        oflag = xpc_dictionary_get_int64( message, "oflag" );
        serviceID = xpc_dictionary_get_uint64( message, "service" );

        service = IOServiceGetMatchingService( kIOMasterPortDefault, IORegistryEntryIDMatching( serviceID ) );

        if ( service )
        {
            fd = _Open( service, oflag, authorizationID );

            if ( fd == -1 )
            {
                status = unix_err( errno );
            }
            else
            {
                status = kIOReturnSuccess;
            }

            IOObjectRelease( service );
        }
        else
        {
            status = unix_err( EINVAL );
        }

        reply = xpc_dictionary_create_reply( message );

        if ( reply )
        {
            xpc_dictionary_set_uint64( reply, "status", status );

            if ( status == kIOReturnSuccess )
            {
               xpc_dictionary_set_fd( reply, "fd", fd );
            }

            xpc_connection_send_message( connection, reply );
        }

        if ( status == kIOReturnSuccess )
        {
            close( fd );
        }
    }
}

static void __ConnectionCallback( xpc_connection_t connection )
{
    pid_t processID;

    processID = xpc_connection_get_pid( connection );

    if ( processID )
    {
        struct proc_uniqidentifierinfo authorizationID = { };

        proc_pidinfo( processID, PROC_PIDUNIQIDENTIFIERINFO, 0, &authorizationID, sizeof( authorizationID ) );

        if ( authorizationID.p_uniqueid )
        {
            xpc_connection_set_event_handler( connection, ^( xpc_object_t message )
            {
                __MessageCallback( connection, message, authorizationID.p_uniqueid );
            } );

            xpc_connection_resume( connection );

            return;
        }
    }

    xpc_connection_cancel( connection );
}

int main( int argc __unused, const char * argv[ ] __unused )
{
    xpc_main( __ConnectionCallback );

    return 0;
}
