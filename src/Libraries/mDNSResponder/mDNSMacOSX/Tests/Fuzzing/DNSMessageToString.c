/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

#include <mdns/DNSMessage.h>

#ifdef FUZZING_DNSMESSAGETOSTRING
int LLVMFuzzerTestOneInput(const char* Data, const size_t Length)
{
    if(Length < 2) {
        return 0;
    }

    int flags = Data[0];
    uint8_t* copy = malloc(Length - 1);
    char* outString = NULL;

    memcpy(copy, Data+1, Length - 1);

    DNSMessageToString(copy, Length - 1, (DNSMessageToStringFlags) flags, &outString);

    free(copy);
    if(outString) {
        free(outString);
    }

    return 0;
}
#endif
