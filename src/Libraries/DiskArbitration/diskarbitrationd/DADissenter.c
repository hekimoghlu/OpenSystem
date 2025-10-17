/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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

DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status )
{
    CFMutableDictionaryRef dissenter;

    dissenter = CFDictionaryCreateMutable( allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );

    if ( dissenter )
    {
        ___CFDictionarySetIntegerValue( dissenter, _kDADissenterStatusKey, status );
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

void DADissenterSetProcessID( DADissenterRef dissenter, pid_t pid )
{
    ___CFDictionarySetIntegerValue( ( void * ) dissenter, _kDADissenterProcessIDKey, pid );
}
