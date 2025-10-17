/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#include <fcntl.h>
#include <errno.h>

#include <pcap/pcap.h>

FILE * outfile = NULL;

static int bufferToFile(const char * name, const uint8_t *Data, size_t Size) {
    FILE * fd;
    if (remove(name) != 0) {
        if (errno != ENOENT) {
            printf("failed remove, errno=%d\n", errno);
            return -1;
        }
    }
    fd = fopen(name, "wb");
    if (fd == NULL) {
        printf("failed open, errno=%d\n", errno);
        return -2;
    }
    if (fwrite (Data, 1, Size, fd) != Size) {
        fclose(fd);
        return -3;
    }
    fclose(fd);
    return 0;
}

void fuzz_openFile(const char * name) {
    if (outfile != NULL) {
        fclose(outfile);
    }
    outfile = fopen(name, "w");
}

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    pcap_t * pkts;
    char errbuf[PCAP_ERRBUF_SIZE];
    const u_char *pkt;
    struct pcap_pkthdr *header;
    struct pcap_stat stats;
    int r;

    //initialize output file
    if (outfile == NULL) {
        outfile = fopen("/dev/null", "w");
        if (outfile == NULL) {
            return 0;
        }
    }

    //rewrite buffer to a file as libpcap does not have buffer inputs
    if (bufferToFile("/tmp/fuzz.pcap", Data, Size) < 0) {
        return 0;
    }

    //initialize structure
    pkts = pcap_open_offline("/tmp/fuzz.pcap", errbuf);
    if (pkts == NULL) {
        fprintf(outfile, "Couldn't open pcap file %s\n", errbuf);
        return 0;
    }

    //loop over packets
    r = pcap_next_ex(pkts, &header, &pkt);
    while (r > 0) {
        //TODO pcap_offline_filter
        fprintf(outfile, "packet length=%d/%d\n",header->caplen, header->len);
        r = pcap_next_ex(pkts, &header, &pkt);
    }
    if (pcap_stats(pkts, &stats) == 0) {
        fprintf(outfile, "number of packets=%d\n", stats.ps_recv);
    }
    //close structure
    pcap_close(pkts);

    return 0;
}
