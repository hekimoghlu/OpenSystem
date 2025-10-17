/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
/* \summary: ZigBee Encapsulation Protocol (ZEP) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"

#include "extract.h"

/* From wireshark packet-zep.c:
 *
 ***********************************************************************
 *
 * ZEP Packets must be received in the following format:
 *
 * |UDP Header|  ZEP Header |IEEE 802.15.4 Packet|
 * | 8 bytes  | 16/32 bytes |    <= 127 bytes    |
 *
 ***********************************************************************
 *
 * ZEP v1 Header will have the following format:
 * |Preamble|Version|Channel ID|Device ID|CRC/LQI Mode|LQI Val|Reserved|Length|
 * |2 bytes |1 byte |  1 byte  | 2 bytes |   1 byte   |1 byte |7 bytes |1 byte|
 *
 * ZEP v2 Header will have the following format (if type=1/Data):
 * |Prmbl|Ver  |Type |ChnlID|DevID|C/L Mode|LQI|NTP TS|Seq#|Res |Len|
 * | 2   | 1   | 1   | 1    | 2   | 1      | 1 | 8    | 4  | 10 | 1 |
 *
 * ZEP v2 Header will have the following format (if type=2/Ack):
 * |Preamble|Version| Type |Sequence#|
 * |2 bytes |1 byte |1 byte| 4 bytes |
 *------------------------------------------------------------
 */

#define     JAN_1970        2208988800U

/* Print timestamp */
static void zep_print_ts(netdissect_options *ndo, const u_char *p)
{
	int32_t i;
	uint32_t uf;
	uint32_t f;
	float ff;

	i = GET_BE_U_4(p);
	uf = GET_BE_U_4(p + 4);
	ff = (float) uf;
	if (ff < 0.0)           /* some compilers are buggy */
		ff += FMAXINT;
	ff = (float) (ff / FMAXINT); /* shift radix point by 32 bits */
	f = (uint32_t) (ff * 1000000000.0);  /* treat fraction as parts per
						billion */
	ND_PRINT("%u.%09d", i, f);

	/*
	 * print the time in human-readable format.
	 */
	if (i) {
		time_t seconds = i - JAN_1970;
		char time_buf[128];

		ND_PRINT(" (%s)",
		    nd_format_time(time_buf, sizeof (time_buf), "%Y/%m/%d %H:%M:%S",
		      localtime(&seconds)));
	}
}

/*
 * Main function to print packets.
 */

void
zep_print(netdissect_options *ndo,
	  const u_char *bp, u_int len)
{
	uint8_t version, inner_len;
	uint32_t seq_no;

	ndo->ndo_protocol = "zep";

	nd_print_protocol_caps(ndo);

	/* Preamble Code (must be "EX") */
	if (GET_U_1(bp) != 'E' || GET_U_1(bp + 1) != 'X') {
		ND_PRINT(" [Preamble Code: ");
		fn_print_char(ndo, GET_U_1(bp));
		fn_print_char(ndo, GET_U_1(bp + 1));
		ND_PRINT("]");
		nd_print_invalid(ndo);
		return;
	}

	version = GET_U_1(bp + 2);
	ND_PRINT("v%u ", version);

	if (version == 1) {
		/* ZEP v1 packet. */
		ND_PRINT("Channel ID %u, Device ID 0x%04x, ",
			 GET_U_1(bp + 3), GET_BE_U_2(bp + 4));
		if (GET_U_1(bp + 6))
			ND_PRINT("CRC, ");
		else
			ND_PRINT("LQI %u, ", GET_U_1(bp + 7));
		inner_len = GET_U_1(bp + 15);
		ND_PRINT("inner len = %u", inner_len);

		bp += 16;
		len -= 16;
	} else {
		/* ZEP v2 packet. */
		if (GET_U_1(bp + 3) == 2) {
			/* ZEP v2 ack. */
			seq_no = GET_BE_U_4(bp + 4);
			ND_PRINT("ACK, seq# = %u", seq_no);
			inner_len = 0;
			bp += 8;
			len -= 8;
		} else {
			/* ZEP v2 data, or some other. */
			ND_PRINT("Type %u, Channel ID %u, Device ID 0x%04x, ",
				 GET_U_1(bp + 3), GET_U_1(bp + 4),
				 GET_BE_U_2(bp + 5));
			if (GET_U_1(bp + 7))
				ND_PRINT("CRC, ");
			else
				ND_PRINT("LQI %u, ", GET_U_1(bp + 8));

			zep_print_ts(ndo, bp + 9);
			seq_no = GET_BE_U_4(bp + 17);
			inner_len = GET_U_1(bp + 31);
			ND_PRINT(", seq# = %u, inner len = %u",
				 seq_no, inner_len);
			bp += 32;
			len -= 32;
		}
	}

	if (inner_len != 0) {
		/* Call 802.15.4 dissector. */
		ND_PRINT("\n\t");
		if (ieee802_15_4_print(ndo, bp, inner_len)) {
			ND_TCHECK_LEN(bp, len);
			bp += len;
			len = 0;
		}
	}

	if (!ndo->ndo_suppress_default_print)
		ND_DEFAULTPRINT(bp, len);
}
