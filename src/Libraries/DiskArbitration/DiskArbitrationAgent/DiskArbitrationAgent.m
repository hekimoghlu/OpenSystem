/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include "DAAgent.h"
#include "DADialog.h"

#include <xpc/xpc.h>
#include <DiskArbitration/DiskArbitrationPrivate.h>
#include <os/log.h>

static void __DAAgentMessageCallback( xpc_object_t object );

static void __DAAgentConnectionCallback( xpc_object_t object )
{
    xpc_type_t type;

    type = xpc_get_type( object );

    if ( type == XPC_TYPE_CONNECTION )
    {
        xpc_connection_set_event_handler( object, ^( xpc_object_t object ) { __DAAgentMessageCallback( object ); } );

        xpc_connection_resume( object );
    }
}

static void __DAAgentMessageCallback( xpc_object_t object )
{
    xpc_type_t type;

    type = xpc_get_type( object );

    if ( type == XPC_TYPE_DICTIONARY )
    {
        xpc_object_t array;
        size_t       _diskSize;

        array = xpc_dictionary_get_value( object, _kDAAgentDiskKey );

        if ( array )
        {
             CFErrorRef errorRef = NULL;
             int count;
             int index;

            if (XPC_TYPE_ARRAY == xpc_get_type( array ))
            {
                count = xpc_array_get_count( array );

                CFMutableArrayRef disklist;
                disklist =  CFArrayCreateMutable( kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks );
                DASessionRef session;

                session = DASessionCreate( kCFAllocatorDefault );
                if ( session )
                {
                    for ( index = 0; index < count; index++ )
                    {
                        xpc_object_t connDict = xpc_array_get_value(array, index);

                        CFDataRef serialization;
                        const void * _disk = xpc_dictionary_get_data( connDict, _kDAAgentDiskKey, &_diskSize );
                        serialization = CFDataCreateWithBytesNoCopy( kCFAllocatorDefault, _disk, _diskSize, kCFAllocatorNull );

                        if ( serialization )
                        {
                       
                            DADiskRef disk;

                            disk = _DADiskCreateFromSerialization( kCFAllocatorDefault, session, serialization );

                            if ( disk )
                            {
                                CFArrayInsertValueAtIndex( disklist, index, disk);

                                CFRelease( disk );
                            }

                            CFRelease( serialization );
                        }
                    }
                    CFRelease( session );
                }
                if ( 0 != CFArrayGetCount(disklist) )
                {
                    _DAAgentAction _action;

                    _action = xpc_dictionary_get_uint64( object, _kDAAgentActionKey );

                    switch ( _action )
                    {
                        case _kDAAgentActionShowDeviceRemoval:
                        {
                            DADialogShowDeviceRemoval( disklist);
                            break;
                        }
                        case _kDAAgentActionShowDeviceUnreadable:
                        {
                            DADiskRef disk = ( DADiskRef ) CFArrayGetValueAtIndex( disklist, 0 );
                            DADialogShowDeviceUnreadable( disk);
                            break;
                        }
                        case _kDAAgentActionShowDeviceUnrepairable:
                        {
                            DADiskRef disk = ( DADiskRef ) CFArrayGetValueAtIndex( disklist, 0 );
                            DADialogShowDeviceUnrepairable( disk);
                            break;
                        }
                    }
                }
                CFArrayRemoveAllValues( disklist );
                CFRelease( disklist );
            }
      }
    }
}

int main( )
{
    xpc_connection_t connection;

    connection = xpc_connection_create_mach_service( _kDAAgentName, NULL, XPC_CONNECTION_MACH_SERVICE_LISTENER );

    if ( connection )
    {
        xpc_connection_set_event_handler( connection, ^( xpc_object_t object ) { __DAAgentConnectionCallback( object ); } );

        xpc_connection_resume( connection );

        dispatch_main( );
    }

    return 0;
}
