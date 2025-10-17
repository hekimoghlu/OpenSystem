/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network_cmds_lib.h"

char *
clean_non_printable(char *str, const size_t len)
{
	size_t i = 0;

	if (str == NULL || len == 0) {
		return str;
	}

	for (i = 0; i < len && str[i] != 0; i++) {
		if (isprint(str[i]) == 0) {
			str[i] = '?';
		}
	}

	return str;
}

void
dump_hex(const unsigned char *ptr, size_t len)
{
	size_t i;

	for (i = 0; i < len; i++) {
		printf("%02x", ptr[i]);
		if (i % 16 == 15) {
			printf("\n");
		} else if (i % 2 == 1) {
			printf(" ");
		}
	}
	if (i % 16 != 0) {
			printf("\n");
	}
}

uint16_t
in_cksum(uint16_t *addr, uint16_t len)
{
	int nleft = len;
	uint16_t *w = addr;
	uint16_t answer;
	uint32_t sum = 0;

	/*
	 *  Our algorithm is simple, using a 32 bit accumulator (sum),
	 *  we add sequential 16 bit words to it, and at the end, fold
	 *  back all the carry bits from the top 16 bits into the lower
	 *  16 bits.
	 */
	while (nleft > 1)  {
		sum += *w++;
		nleft -= 2;
	}

	/* mop up an odd byte, if necessary */
	if (nleft == 1) {
		sum += *(uint8_t *)w;
	}
	/*
	 * add back carry outs from top 16 bits to low 16 bits
	 */
	sum = (sum >> 16) + (sum & 0xffff);	/* add hi 16 to low 16 */
	sum += (sum >> 16);			/* add carry */
	answer = ~sum;				/* truncate to 16 bits */
	return (answer);
}
