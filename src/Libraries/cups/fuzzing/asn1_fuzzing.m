/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

#include <Foundation/Foundation.h>

#include "config.h"
#include "array.c"
#include "ipp-support.c"
#include "options.c"
#include "transcode.c"
#include "usersys.c"
#include "language.c"
#include "thread.c"
#include "ipp.c"
#include "globals.c"
#include "debug.c"
#include "file.c"
#include "dir.c"
#include "snmp.c"

#include "stubs.m"

int _asn1_fuzzing(Boolean verbose, const uint8_t* data, size_t len)
{
    int save = gVerbose;
    gVerbose = verbose;

    cups_snmp_t packet;
    bzero(&packet, sizeof(packet));

    uint8_t* localData = (uint8_t*) malloc(len);
    memmove(localData, data, len);
    asn1_decode_snmp(localData, len, &packet);
    Boolean matches = memcmp(localData, data, len) == 0;
    free(localData);

    gVerbose = save;

    if (! matches) {
        NSLog(@"asn1_decode_snmp mutated input buffer!");
        return 1;
    }

    return 0;
}

extern int LLVMFuzzerTestOneInput(const uint8_t *buffer, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *buffer, size_t size)
{
    return _asn1_fuzzing(false, buffer, size);
}
