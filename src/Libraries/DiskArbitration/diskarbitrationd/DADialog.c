/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "DADialog.h"

#include "DAAgent.h"
#include "DAMain.h"
#include "DASupport.h"

#include <xpc/private.h>
#include "DALog.h"

static void __DADialogShow( CFMutableArrayRef diskinfoarray, _DAAgentAction action )
{
#if TARGET_OS_OSX
    xpc_object_t message;

    message = xpc_dictionary_create( NULL, NULL, 0 );

    if ( message )
    {
        xpc_connection_t connection;

        connection = xpc_connection_create_mach_service( _kDAAgentName, NULL, 0 );

        if ( connection )
        {
            xpc_dictionary_set_uint64( message, _kDAAgentActionKey, action );

            xpc_object_t array = xpc_array_create(  NULL, 0 );
            int count;
            CFIndex index;

            count = CFArrayGetCount( diskinfoarray );

            for ( index = 0; index < count; index++ )
            {
                xpc_object_t dict = xpc_dictionary_create( NULL, NULL, 0 );
                CFDataRef serialization =  CFArrayGetValueAtIndex( diskinfoarray, index );
                xpc_dictionary_set_data( dict, _kDAAgentDiskKey, CFDataGetBytePtr( serialization ), CFDataGetLength( serialization ) );
                xpc_array_append_value(array, dict);
                xpc_release(dict);
            }

            CFArrayRemoveAllValues( diskinfoarray );
            xpc_dictionary_set_value(message, _kDAAgentDiskKey, array);

            xpc_release(array);

            xpc_connection_set_event_handler( connection, ^( xpc_object_t object ) { } );

            xpc_connection_set_target_uid( connection, gDAConsoleUserUID );

            xpc_connection_resume( connection );

            xpc_connection_send_message( connection, message );

            xpc_release( connection );
        }

        xpc_release( message );
    }
#endif
}

static CFBooleanRef IsNotificationDisabled( CFStringRef preference)
{
    CFBooleanRef value = CFDictionaryGetValue( gDAPreferenceList, preference );

    value = value ? value : kCFBooleanFalse;
    
    return value;
    
}

void DADialogShowDeviceRemoval( CFMutableArrayRef diskinfoarray )
{
#if TARGET_OS_OSX
    if ( kCFBooleanFalse == IsNotificationDisabled ( kDAPreferenceDisableEjectNotificationKey ) )
    {
        __DADialogShow( diskinfoarray, _kDAAgentActionShowDeviceRemoval );
    }
#endif
}

void DADialogShowDeviceUnreadable( DADiskRef disk )
{
 #if TARGET_OS_OSX   
    if ( kCFBooleanFalse == IsNotificationDisabled ( kDAPreferenceDisableUnreadableNotificationKey ) )
    {
        CFMutableArrayRef diskInfoArray = CFArrayCreateMutable( kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks );
        CFDataRef serialization = DADiskGetSerialization( disk );
        CFArrayAppendValue( diskInfoArray, serialization );
        __DADialogShow( diskInfoArray, _kDAAgentActionShowDeviceUnreadable );
        CFRelease( diskInfoArray );
    }
 #endif
    DALogError( "disk is not readable %@", disk);
}

void DADialogShowDeviceUnrepairable( DADiskRef disk )
{
 #if TARGET_OS_OSX 
    if ( kCFBooleanFalse == IsNotificationDisabled ( kDAPreferenceDisableUnrepairableNotificationKey ) )
    {
        CFMutableArrayRef diskInfoArray = CFArrayCreateMutable( kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks );
        CFDataRef serialization = DADiskGetSerialization( disk );
        CFArrayAppendValue( diskInfoArray, serialization );
        __DADialogShow( diskInfoArray, _kDAAgentActionShowDeviceUnrepairable );
        CFRelease( diskInfoArray );
    }
 #endif
    DALogError( "disk is not repairable %@", disk);
}

