/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
/* \summary: Trivial File Transfer Protocol (TFTP) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

/*
 * Trivial File Transfer Protocol (IEN-133)
 */

/*
 * Packet types.
 */
#define	RRQ	01			/* read request */
#define	WRQ	02			/* write request */
#define	DATA	03			/* data packet */
#define	ACK	04			/* acknowledgement */
#define	TFTP_ERROR	05			/* error code */
#define OACK	06			/* option acknowledgement */

/*
 * Error codes.
 */
#define	EUNDEF		0		/* not defined */
#define	ENOTFOUND	1		/* file not found */
#define	EACCESS		2		/* access violation */
#define	ENOSPACE	3		/* disk full or allocation exceeded */
#define	EBADOP		4		/* illegal TFTP operation */
#define	EBADID		5		/* unknown transfer ID */
#define	EEXISTS		6		/* file already exists */
#define	ENOUSER		7		/* no such user */


/* op code to string mapping */
static const struct tok op2str[] = {
	{ RRQ,		"RRQ" },	/* read request */
	{ WRQ,		"WRQ" },	/* write request */
	{ DATA,		"DATA" },	/* data packet */
	{ ACK,		"ACK" },	/* acknowledgement */
	{ TFTP_ERROR,	"ERROR" },	/* error code */
	{ OACK,		"OACK" },	/* option acknowledgement */
	{ 0,		NULL }
};

/* error code to string mapping */
static const struct tok err2str[] = {
	{ EUNDEF,	"EUNDEF" },	/* not defined */
	{ ENOTFOUND,	"ENOTFOUND" },	/* file not found */
	{ EACCESS,	"EACCESS" },	/* access violation */
	{ ENOSPACE,	"ENOSPACE" },	/* disk full or allocation exceeded */
	{ EBADOP,	"EBADOP" },	/* illegal TFTP operation */
	{ EBADID,	"EBADID" },	/* unknown transfer ID */
	{ EEXISTS,	"EEXISTS" },	/* file already exists */
	{ ENOUSER,	"ENOUSER" },	/* no such user */
	{ 0,		NULL }
};

/*
 * Print trivial file transfer program requests
 */
void
tftp_print(netdissect_options *ndo,
           const u_char *bp, u_int length)
{
	const char *cp;
	u_int opcode;
	u_int ui;

	ndo->ndo_protocol = "tftp";

	/* Print protocol */
	nd_print_protocol_caps(ndo);
	/* Print length */
	ND_PRINT(", length %u", length);

	/* Print tftp request type */
	if (length < 2)
		goto trunc;
	opcode = GET_BE_U_2(bp);
	cp = tok2str(op2str, "tftp-#%u", opcode);
	ND_PRINT(", %s", cp);
	/* Bail if bogus opcode */
	if (*cp == 't')
		return;
	bp += 2;
	length -= 2;

	switch (opcode) {

	case RRQ:
	case WRQ:
		if (length == 0)
			goto trunc;
		ND_PRINT(" ");
		/* Print filename */
		ND_PRINT("\"");
		ui = nd_printztn(ndo, bp, length, ndo->ndo_snapend);
		ND_PRINT("\"");
		if (ui == 0)
			goto trunc;
		bp += ui;
		length -= ui;

		/* Print the mode - RRQ and WRQ only */
		if (length == 0)
			goto trunc;	/* no mode */
		ND_PRINT(" ");
		ui = nd_printztn(ndo, bp, length, ndo->ndo_snapend);
		if (ui == 0)
			goto trunc;
		bp += ui;
		length -= ui;

		/* Print options, if any */
		while (length != 0) {
			if (GET_U_1(bp) != '\0')
				ND_PRINT(" ");
			ui = nd_printztn(ndo, bp, length, ndo->ndo_snapend);
			if (ui == 0)
				goto trunc;
			bp += ui;
			length -= ui;
		}
		break;

	case OACK:
		/* Print options */
		while (length != 0) {
			if (GET_U_1(bp) != '\0')
				ND_PRINT(" ");
			ui = nd_printztn(ndo, bp, length, ndo->ndo_snapend);
			if (ui == 0)
				goto trunc;
			bp += ui;
			length -= ui;
		}
		break;

	case ACK:
	case DATA:
		if (length < 2)
			goto trunc;	/* no block number */
		ND_PRINT(" block %u", GET_BE_U_2(bp));
		break;

	case TFTP_ERROR:
		/* Print error code string */
		if (length < 2)
			goto trunc;	/* no error code */
		ND_PRINT(" %s", tok2str(err2str, "tftp-err-#%u \"",
				       GET_BE_U_2(bp)));
		bp += 2;
		length -= 2;
		/* Print error message string */
		if (length == 0)
			goto trunc;	/* no error message */
		ND_PRINT(" \"");
		ui = nd_printztn(ndo, bp, length, ndo->ndo_snapend);
		ND_PRINT("\"");
		if (ui == 0)
			goto trunc;
		break;

	default:
		/* We shouldn't get here */
		ND_PRINT("(unknown #%u)", opcode);
		break;
	}
	return;
trunc:
	nd_print_trunc(ndo);
}
