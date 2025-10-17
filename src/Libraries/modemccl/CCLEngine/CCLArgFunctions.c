/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
/*
 *  CCLArgFunctions.c
 *  CCLLauncher
 *
 *  Created by kevine on 4/11/06.
 *  Copyright 2006-7 Apple, Inc.  All rights reserved.
 *
 */

#include "CCLArgFunctions.h"
#include <CoreFoundation/CoreFoundation.h>

// For compatibility with previous usage, the boolean-return Get*FromDict()
// functions return errors only if the key's value is non-NULL but cannot
// be extracted.  If no value is stored for key or the value is NULL, the
// by-reference return buffers are not modified.

CFDictionaryRef
GetCFDictionaryFromDict(CFDictionaryRef dict, const CFStringRef key)
{
    CFDictionaryRef retVal= (CFDictionaryRef) CFDictionaryGetValue(dict, key);
    if((retVal!= NULL) && (CFDictionaryGetTypeID()== CFGetTypeID(retVal)))
    {
        return retVal;
    }
    return NULL;
}

bool
GetCFStringFromDict(CFDictionaryRef dict, CFStringRef *s, const CFStringRef key)
{
    CFStringRef tempString= (CFStringRef) CFDictionaryGetValue(dict, key);

    if (tempString == NULL)
        return true;

    if (CFStringGetTypeID() == CFGetTypeID(tempString)) {
        *s = tempString;
        return true;
    }

    return false;
}

bool
CopyCStringFromDict(CFDictionaryRef dict, char** string, const CFStringRef key)
{
    CFStringRef str = NULL;
    CFIndex bufSize;
    char *buf = NULL;

    // fallout cases
    if (!GetCFStringFromDict(dict, &str, key))
        return false;
    if (str == NULL)
        return true;

    bufSize = CFStringGetMaximumSizeForEncoding(CFStringGetLength(str),
                                                kCFStringEncodingUTF8) + 1;
    buf = malloc(bufSize);
    if(buf && CFStringGetCString(str, buf, bufSize, kCFStringEncodingUTF8)) {
        *string = buf;
        return true;
    } 

    if (buf)  free(buf);
    return false;
}

bool
GetCFNumberFromDict(CFDictionaryRef dict, CFNumberRef *n, const CFStringRef key)
{
    CFNumberRef tempNum = (CFNumberRef)CFDictionaryGetValue(dict, key);

    if (tempNum == NULL)
        return true;

    if(CFNumberGetTypeID() == CFGetTypeID(tempNum)) {
        *n = tempNum;
        return true;
    }

    return false;
}

bool
GetIntFromDict(CFDictionaryRef dict, int* intRef, const CFStringRef key)
{
    CFNumberRef num = NULL;

    // fallout cases
    if (!GetCFNumberFromDict(dict, &num, key))
        return false;
    if (num == NULL)
        return true;

    return CFNumberGetValue(num, kCFNumberSInt32Type, intRef);
}
