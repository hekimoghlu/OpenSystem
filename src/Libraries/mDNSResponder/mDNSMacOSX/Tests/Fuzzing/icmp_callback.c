/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#include "ioloop.h"
#include "srp-gw.h"
#include "srp-proxy.h"
#include "srp-mdns-proxy.h"
#include "route.h"

#ifdef FUZZING_ICMP_CALLBACK

void icmp_callback(io_t *io, void *context);
extern interface_t *interfaces;
extern char *thread_interface_name;
interface_t interface = {0};

void hexdump(const char* Data, ssize_t Length) {
    for(ssize_t i = 0; i < Length; i++) {
        dprintf(2, "%02x", (unsigned char) Data[i]);
    }
    dprintf(2, "\n");
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    interfaces = &interface;
    interface.next = 0;
    interface.name = "fuzz0";
    // interface.deconfigure_wakeup = calloc(sizeof(interface.deconfigure_wakeup), 1);

    thread_interface_name = "fuzz0";

    return 0;
}


int LLVMFuzzerTestOneInput(const char* Data, const size_t Length)
{
    io_t io = {0};
    ssize_t written = 0;

    // Minimum size for ICMP header
    if(Length < 8) {
        return 0;
    }

    // hexdump(Data, Length);

    int filedes[2];
    if(pipe(filedes)) {
        perror("pipe(2) failed!");
        return 1;
    }

    written = write(filedes[1], Data, Length);
    if(written <= 0) {
        return 1;
    }

    io.fd = filedes[0];
    icmp_callback(&io, 0);

    close(filedes[0]);
    close(filedes[1]);

    return 0;
}
#endif
