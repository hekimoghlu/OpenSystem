/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "DACallback.h"

DACallbackRef DACallbackCreate( CFAllocatorRef   allocator,
                                DASessionRef     session,
                                mach_vm_offset_t address,
                                mach_vm_offset_t context,
                                _DACallbackKind  kind,
                                CFIndex          order,
                                CFDictionaryRef  match,
                                CFArrayRef       watch )
{
    CFMutableDictionaryRef callback;

    callback = CFDictionaryCreateMutable( allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );

    if ( callback )
    {
        CFDictionarySetValue( callback, _kDACallbackSessionKey, session );

        ___CFDictionarySetIntegerValue( callback, _kDACallbackAddressKey, address );
        ___CFDictionarySetIntegerValue( callback, _kDACallbackContextKey, context );
        ___CFDictionarySetIntegerValue( callback, _kDACallbackKindKey,    kind    );
        ___CFDictionarySetIntegerValue( callback, _kDACallbackOrderKey,   order   );

        if ( match )  CFDictionarySetValue( callback, _kDACallbackMatchKey, match );
        if ( watch )  CFDictionarySetValue( callback, _kDACallbackWatchKey, watch );
    }

    return ( void * ) callback;
}

DACallbackRef DACallbackCreateCopy( CFAllocatorRef allocator, DACallbackRef callback )
{
    return ( void * ) CFDictionaryCreateMutableCopy( allocator, 0, ( void * ) callback );
}

mach_vm_offset_t DACallbackGetAddress( DACallbackRef callback )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) callback, _kDACallbackAddressKey );
}

CFTypeRef DACallbackGetArgument0( DACallbackRef callback )
{
    return CFDictionaryGetValue( ( void * ) callback, _kDACallbackArgument0Key );
}

CFTypeRef DACallbackGetArgument1( DACallbackRef callback )
{
    return CFDictionaryGetValue( ( void * ) callback, _kDACallbackArgument1Key );
}

mach_vm_offset_t DACallbackGetContext( DACallbackRef callback )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) callback, _kDACallbackContextKey );
}

DADiskRef DACallbackGetDisk( DACallbackRef callback )
{
    return ( void * ) CFDictionaryGetValue( ( void * ) callback, _kDACallbackDiskKey );
}

_DACallbackKind DACallbackGetKind( DACallbackRef callback )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) callback, _kDACallbackKindKey );
}

CFDictionaryRef DACallbackGetMatch( DACallbackRef callback )
{
    return CFDictionaryGetValue( ( void * ) callback, _kDACallbackMatchKey );
}

SInt32 DACallbackGetOrder( DACallbackRef callback )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) callback, _kDACallbackOrderKey );
}

DASessionRef DACallbackGetSession( DACallbackRef callback )
{
    return ( void * ) CFDictionaryGetValue( ( void * ) callback, _kDACallbackSessionKey );
}

CFAbsoluteTime DACallbackGetTime( DACallbackRef callback )
{
    CFDateRef      date;
    CFAbsoluteTime time = 0;

    date = CFDictionaryGetValue( ( void * ) callback, _kDACallbackTimeKey );

    if ( date )
    {
        time = CFDateGetAbsoluteTime( date );
    }
    
    return time;
}

CFArrayRef DACallbackGetWatch( DACallbackRef callback )
{
    return CFDictionaryGetValue( ( void * ) callback, _kDACallbackWatchKey );
}

void DACallbackSetArgument0( DACallbackRef callback, CFTypeRef argument0 )
{
    if ( argument0 )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackArgument0Key, argument0 );
    }
    else
    {
        CFDictionaryRemoveValue( ( void * ) callback, _kDACallbackArgument0Key );
    }
}

void DACallbackSetArgument1( DACallbackRef callback, CFTypeRef argument1 )
{
    if ( argument1 )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackArgument1Key, argument1 );
    }
    else
    {
        CFDictionaryRemoveValue( ( void * ) callback, _kDACallbackArgument1Key );
    }
}

void DACallbackSetDisk( DACallbackRef callback, DADiskRef disk )
{
    if ( disk )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackDiskKey, disk );
    }
    else
    {
        CFDictionaryRemoveValue( ( void * ) callback, _kDACallbackDiskKey );
    }
}

void DACallbackSetMatch( DACallbackRef callback, CFDictionaryRef match )
{
    if ( match )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackMatchKey, match );
    }
    else
    {
        CFDictionaryRemoveValue( ( void * ) callback, _kDACallbackMatchKey );
    }
}

void DACallbackSetSession( DACallbackRef callback, DASessionRef session )
{
    if ( session )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackSessionKey, session );
    }
    else
    {
        CFDictionaryRemoveValue( ( void * ) callback, _kDACallbackSessionKey );
    }
}

void DACallbackSetTime( DACallbackRef callback, CFAbsoluteTime time )
{
    CFDateRef date;

    date = CFDateCreate( kCFAllocatorDefault, time );

    if ( date )
    {
        CFDictionarySetValue( ( void * ) callback, _kDACallbackTimeKey, date );

        CFRelease( date );
    }
}
