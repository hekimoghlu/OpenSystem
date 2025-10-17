/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

//
//  SOSGenCount.c
//  sec
//
//  Created by Richard Murphy on 1/29/15.
//
//

#include "SOSGenCount.h"
#include <utilities/SecCFWrappers.h>
#include <CoreFoundation/CFLocale.h>
#include <utilities/der_plist.h>
#include <utilities/der_plist_internal.h>


static CFAbsoluteTime SOSGenerationCountGetDateBits(int64_t genValue) {
    return (uint32_t) ((((uint64_t) genValue) >> 32) << 1);
}

void SOSGenerationCountWithDescription(SOSGenCountRef gen, void (^operation)(CFStringRef description)) {
    CFStringRef description = SOSGenerationCountCopyDescription(gen);

    operation(description);

    CFReleaseSafe(description);
}

CFStringRef SOSGenerationCountCopyDescription(SOSGenCountRef gen) {
    CFMutableStringRef gcDecsription = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, CFSTR("["));
    if(gen) {
        int64_t value = SOSGetGenerationSint(gen);
        withStringOfAbsoluteTime(SOSGenerationCountGetDateBits(value), ^(CFStringRef decription) {
            CFStringAppend(gcDecsription, decription);
        });
        CFStringAppendFormat(gcDecsription, NULL, CFSTR(" %u]"), (uint32_t)(value & 0xFFFFFFFF));
    } else {
        CFStringAppend(gcDecsription, CFSTR("NA]"));
    }
    return gcDecsription;
}

int64_t SOSGetGenerationSint(SOSGenCountRef gen) {
    int64_t value;
    if(!gen) return 0;
    CFNumberGetValue(gen, kCFNumberSInt64Type, &value);
    return value;
}

static int64_t sosGenerationSetHighBits(int64_t value, int32_t high_31)
{
    value &= 0xFFFFFFFF; // Keep the low 32 bits.
    value |= ((int64_t) high_31) << 32;

    return value;
}

static SOSGenCountRef sosGenerationCreateOrIncrement(SOSGenCountRef gen) {
    int64_t value = 0;

    if(gen) {
        value = SOSGetGenerationSint(gen);
    }

    if((value >> 32) == 0) {
        uint32_t seconds = CFAbsoluteTimeGetCurrent(); // seconds
        value = sosGenerationSetHighBits(value, (seconds >> 1));
    }

    value++;
    return CFNumberCreate(NULL, kCFNumberSInt64Type, &value);
}

SOSGenCountRef SOSGenerationCreate(void) {
    return sosGenerationCreateOrIncrement(NULL);
}


// We need this for a circle gencount test
SOSGenCountRef SOSGenerationCreateWithValue(int64_t value) {
    return CFNumberCreate(NULL, kCFNumberSInt64Type, &value);
}

SOSGenCountRef SOSGenerationIncrementAndCreate(SOSGenCountRef gen) {
    return sosGenerationCreateOrIncrement(gen);
}

SOSGenCountRef SOSGenerationCopy(SOSGenCountRef gen) {
    if(!gen) return NULL;
    int64_t value = SOSGetGenerationSint(gen);
    return CFNumberCreate(NULL, kCFNumberSInt64Type, &value);
}

// Is current older than proposed?
bool SOSGenerationIsOlder(SOSGenCountRef older, SOSGenCountRef newer) {
    switch(CFNumberCompare(older, newer, NULL)) {
        case kCFCompareLessThan:  return true;
        case kCFCompareEqualTo: return false;
        case kCFCompareGreaterThan:  return false;
    }
    return false;
}

// Is current older than proposed?
static bool SOSGenerationIsOlderOrEqual(SOSGenCountRef older, SOSGenCountRef newer) {
    switch(CFNumberCompare(older, newer, NULL)) {
        case kCFCompareLessThan:  return true;
        case kCFCompareEqualTo: return true;
        case kCFCompareGreaterThan:  return false;
    }
    return false;
}


SOSGenCountRef SOSGenerationCreateWithBaseline(SOSGenCountRef reference) {
    SOSGenCountRef retval = SOSGenerationCreate();
    if(SOSGenerationIsOlderOrEqual(retval, reference)) {
        CFReleaseNull(retval);
        retval = SOSGenerationIncrementAndCreate(reference);
    }
    return retval;
}


SOSGenCountRef SOSGenCountCreateFromDER(CFAllocatorRef allocator, CFErrorRef* error,
                                        const uint8_t** der_p, const uint8_t *der_end) {
    SOSGenCountRef retval = NULL;
    *der_p = der_decode_number(allocator, &retval, error, *der_p, der_end);
    if(retval == NULL)
        retval = SOSGenerationCreate();
    return retval;
}

size_t SOSGenCountGetDEREncodedSize(SOSGenCountRef gencount, CFErrorRef *error) {
    return der_sizeof_number(gencount, error);
}

uint8_t *SOSGenCountEncodeToDER(SOSGenCountRef gencount, CFErrorRef* error, const uint8_t* der, uint8_t* der_end) {
    return der_encode_number(gencount, error, der, der_end);
}



