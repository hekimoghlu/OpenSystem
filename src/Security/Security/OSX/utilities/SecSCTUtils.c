/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
//  SecSCTUtils.c
//  utilities
/*
 * Copyright (c) 2015 Apple Inc. All Rights Reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 *
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 *
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 *
 * @APPLE_LICENSE_HEADER_END@
 */

#include <AssertMacros.h>
#include <utilities/SecCFWrappers.h>
#include "SecSCTUtils.h"

static size_t SSLDecodeSize(const uint8_t *p)
{
    return (p[0]<<8 | p[1]);
}

CFArrayRef SecCreateSignedCertificateTimestampsArrayFromSerializedSCTList(const uint8_t *p, size_t listLen)
{
    size_t encodedListLen;
    CFMutableArrayRef sctArray = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
    require_quiet(sctArray, out);

    require(listLen > 2 , out);
    encodedListLen = SSLDecodeSize(p); p+=2; listLen-=2;

    require(encodedListLen==listLen, out);

    while (listLen > 0)
    {
        size_t itemLen;
        require(listLen >= 2, out);
        itemLen = SSLDecodeSize(p); p += 2; listLen-=2;
        require(itemLen <= listLen, out);
        CFDataRef sctData = CFDataCreate(kCFAllocatorDefault, p, itemLen);
        p += itemLen; listLen -= itemLen;
        require(sctData, out);
        CFArrayAppendValue(sctArray, sctData);
        CFReleaseSafe(sctData);
    }

    return sctArray;

out:
    CFReleaseSafe(sctArray);
    return NULL;
}
