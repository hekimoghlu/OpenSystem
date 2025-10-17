/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "misc_util.h"

/*********************************************************************
*********************************************************************/
char * createUTF8CStringForCFString(CFStringRef aString)
{
    char * result = NULL;
    CFIndex bufferLength = 0;

    if (!aString) {
        goto finish;
    }

    bufferLength = sizeof('\0') +
        CFStringGetMaximumSizeForEncoding(CFStringGetLength(aString),
	    kCFStringEncodingUTF8);

    result = (char *)malloc(bufferLength * sizeof(char));
    if (!result) {
        goto finish;
    }
    if (!CFStringGetCString(aString, result, bufferLength,
        kCFStringEncodingUTF8)) {

        SAFE_FREE_NULL(result);
        goto finish;
    }

finish:
    return result;
}

/*********************************************************************
*********************************************************************/
CFStringRef createCFStringForData(CFDataRef aData, CFIndex maxBytes)
{
    CFMutableStringRef  result = NULL;
    const uint8_t     * bytes  = NULL;  // do not free
    CFIndex            count, i;
    
    result = CFStringCreateMutable(kCFAllocatorDefault, /* maxLength */ 0);
    if (!result) {
        goto finish;
    }
    
    count = CFDataGetLength(aData);

    CFStringAppend(result, CFSTR("<"));

    if (count) {
        bytes = CFDataGetBytePtr(aData);;
        for (i = 0; i < count && i < maxBytes; i++) {
            CFStringAppendFormat(result, /* options */ NULL, CFSTR("%02x%s"),
                (unsigned)(bytes[i]),
                (i > 0 && !((i + 1) % 4) && (i + 1 < count)) ? " " : "");
        }
        if (maxBytes < count) {
            CFStringAppendFormat(result, /* options */ NULL,
                CFSTR("...(%u bytes total)"), (unsigned)count);
        }
    }
    CFStringAppend(result, CFSTR(">"));

finish:
    return result;
}
