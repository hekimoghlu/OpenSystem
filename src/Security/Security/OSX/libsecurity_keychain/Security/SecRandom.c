/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
 * SecFramework.c - generic non API class specific functions
 */

#define SEC_BUILDER 1

//#include "SecFramework.h"
#include <strings.h>
#include <CoreFoundation/CFBundle.h>
#include <CoreFoundation/CFURLAccess.h>
#include "SecRandomP.h"
#include <CommonCrypto/CommonRandomSPI.h>
#include <stdlib.h>

/* Default random ref for /dev/random. */
const SecRandomRef kSecRandomDefault = NULL;


int SecRandomCopyBytes(__unused SecRandomRef rnd, size_t count, void *bytes) {
    return CCRandomCopyBytes(kCCRandomDefault, bytes, count);
}


CFDataRef
SecRandomCopyData(SecRandomRef rnd, size_t count)
{
    uint8_t *bytes;
    CFDataRef retval = NULL;
    
    if (rnd != kSecRandomDefault) return NULL;
    if((bytes = malloc(count)) == NULL) return NULL;
    if(CCRandomCopyBytes(kCCRandomDefault, bytes, count) == kCCSuccess)
        retval = CFDataCreate(kCFAllocatorDefault, bytes, count);
    bzero(bytes, count);
    free(bytes);
    return retval;
}

