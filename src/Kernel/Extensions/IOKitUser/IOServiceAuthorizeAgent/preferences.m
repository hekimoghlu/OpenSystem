/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#include "preferences.h"

dispatch_queue_t __PreferencesGetQueue( void )
{
    static dispatch_once_t once;
    static dispatch_queue_t queue;

    dispatch_once( &once, ^
    {
        queue = dispatch_queue_create( 0, 0 );
    } );

    return queue;
}

void _PreferencesAppendArrayValue( CFStringRef key, CFPropertyListRef value )
{
    dispatch_sync( __PreferencesGetQueue( ), ^
    {
        CFArrayRef array;
        CFMutableArrayRef mutableArray;

        array = _PreferencesCopyValue( key );

        if ( array )
        {
            mutableArray = CFArrayCreateMutableCopy( kCFAllocatorDefault, 0, array );

            CFRelease( array );
        }
        else
        {
            mutableArray = CFArrayCreateMutable( kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks );
        }

        if ( mutableArray )
        {
            CFArrayAppendValue( mutableArray, value );

            _PreferencesSetValue( key, mutableArray );

            CFRelease( mutableArray );
        }
    } );
}

CFPropertyListRef _PreferencesCopyValue( CFStringRef key )
{
    return CFPreferencesCopyAppValue( key, kCFPreferencesCurrentApplication );
}

void _PreferencesSetValue( CFStringRef key, CFPropertyListRef value )
{
    CFPreferencesSetAppValue( key, value, kCFPreferencesCurrentApplication );

    CFPreferencesAppSynchronize( kCFPreferencesCurrentApplication );
}

