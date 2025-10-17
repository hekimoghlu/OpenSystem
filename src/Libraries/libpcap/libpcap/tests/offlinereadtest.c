/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include <err.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <net/bpf.h>
#include <pcap.h>


#define PAD32(x) (((x) + 3) & ~3)
#define	SWAPLONG(y) \
((((y)&0xff)<<24) | (((y)&0xff00)<<8) | (((y)&0xff0000)>>8) | (((y)>>24)&0xff))
#define	SWAPSHORT(y) \
( (((y)&0xff)<<8) | ((u_short)((y)&0xff00)>>8) )
#define	SWAPLONGLONG(y) \
(SWAPLONG((unsigned long)(y)) << 32 | SWAPLONG((unsigned long)((y) >> 32)))

void hex_and_ascii_print(const char *, const void *, size_t, const char *);

void
read_callback(u_char *user, const struct pcap_pkthdr *hdr, const u_char *bytes)
{
	fprintf(stderr, "pcap_pkthdr ts %ld.%06d caplen %u len %u\n",
			hdr->ts.tv_sec,
			hdr->ts.tv_usec,
			hdr->caplen,
			hdr->len);

	hex_and_ascii_print("", bytes, hdr->caplen, "\n");
}

int
main(int argc, const char * argv[])
{
	int i;
	char errbuf[PCAP_ERRBUF_SIZE];

	for (i = 1; i < argc; i++) {
		pcap_t *pcap;

		if (strcmp(argv[i], "-h") == 0) {
			char *path = strdup((argv[0]));
			printf("# usage: %s  file...\n", getprogname());
			if (path != NULL)
				free(path);
			exit(0);
		}

		printf("#\n# opening %s\n#\n", argv[i]);

		pcap = pcap_open_offline(argv[i], errbuf);
		if (pcap == NULL) {
			warnx("pcap_open_offline(%s) failed: %s\n",
				  argv[i], errbuf);
			continue;
		}
		printf("datalink %d\n", pcap_datalink(pcap));
		struct bpf_program fcode = {};
		if (pcap_compile(pcap, &fcode, "", 1, 0) < 0)
			warnx("%s", pcap_geterr(pcap));

		int result = pcap_loop(pcap, -1, read_callback, (u_char *)pcap);
		if (result < 0) {
			warnx("pcap_dispatch failed: %s\n",
				  pcap_statustostr(result));
		} else {
			printf("# read %d packets\n", result);
		}
		pcap_close(pcap);
	}
	return 0;
}
