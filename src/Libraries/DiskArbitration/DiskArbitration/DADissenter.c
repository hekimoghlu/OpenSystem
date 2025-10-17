/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#include "DADissenter.h"

#include "DAInternal.h"

#include "DiskArbitrationPrivate.h"
#include <unistd.h>

DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string )
{
    CFMutableDictionaryRef dissenter;

    dissenter = CFDictionaryCreateMutable( allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );

    if ( dissenter )
    {
        ___CFDictionarySetIntegerValue( dissenter, _kDADissenterProcessIDKey, getpid( ) );
        ___CFDictionarySetIntegerValue( dissenter, _kDADissenterStatusKey,    status    );

        if ( string )
        {
            CFDictionarySetValue( dissenter, _kDADissenterStatusStringKey, string );
        }
    }

    return ( void * ) dissenter;
}
 
pid_t DADissenterGetProcessID( DADissenterRef dissenter )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) dissenter, _kDADissenterProcessIDKey );
}

DAReturn DADissenterGetStatus( DADissenterRef dissenter )
{
    return ___CFDictionaryGetIntegerValue( ( void * ) dissenter, _kDADissenterStatusKey );
}

CFStringRef DADissenterGetStatusString( DADissenterRef dissenter )
{
    return CFDictionaryGetValue( ( void * ) dissenter, _kDADissenterStatusStringKey );
}
