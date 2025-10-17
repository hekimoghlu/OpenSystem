/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#include "srp.h"
#include "dns-msg.h"

#ifdef FUZZING_DNS_WIRE_PARSE
int LLVMFuzzerTestOneInput(const char* Data, const size_t Length)
{
    dns_message_t *message = 0;
    dns_wire_t wire = {0};
    unsigned len = (unsigned) Length;

    // At least one byte of data is needed
    if (Length < 1 + __builtin_offsetof(dns_wire_t, data))
        return 0;

    // Too much data
    if (Length > sizeof(wire))
        return 0;

    // Initialize the wire struct with the fuzzing data
    memcpy(&wire, Data, Length);

    // Parse!
    dns_wire_parse(&message, &wire, len);

    if(message)
        free(message);

    return 0;
}
#endif
