/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include <stdlib.h>
#include <string.h>

#include <pcap/pcap.h>

void fuzz_openFile(const char * name){
    //do nothing
}

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    pcap_t * pkts;
    struct bpf_program bpf;
    char * filter;

    //we need at least 1 byte for linktype
    if (Size < 1) {
        return 0;
    }

    //initialize structure snaplen = 65535
    pkts = pcap_open_dead(Data[Size-1], 0xFFFF);
    if (pkts == NULL) {
        printf("pcap_open_dead failed\n");
        return 0;
    }
    filter = malloc(Size);
    memcpy(filter, Data, Size);
    //null terminate string
    filter[Size-1] = 0;

    if (pcap_compile(pkts, &bpf, filter, 1, PCAP_NETMASK_UNKNOWN) == 0) {
        pcap_setfilter(pkts, &bpf);
        pcap_close(pkts);
        pcap_freecode(&bpf);
    }
    else {
        pcap_close(pkts);
    }
    free(filter);

    return 0;
}
