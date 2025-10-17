/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
/* \summary: Kerberos printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

/*
 * Kerberos 4:
 *
 * Athena Technical Plan
 * Section E.2.1
 * Kerberos Authentication and Authorization System
 * by S. P. Miller, B. C. Neuman, J. I. Schiller, and J. H. Saltzer
 *
 * https://web.mit.edu/Saltzer/www/publications/athenaplan/e.2.1.pdf
 *
 * 7. Appendix I Design Specifications
 *
 * Kerberos 5:
 *
 * RFC 1510, RFC 2630, etc.
 */


static const u_char *c_print(netdissect_options *, const u_char *, const u_char *);
static const u_char *krb4_print_hdr(netdissect_options *, const u_char *);
static void krb4_print(netdissect_options *, const u_char *);

#define AUTH_MSG_KDC_REQUEST			1<<1
#define AUTH_MSG_KDC_REPLY			2<<1
#define AUTH_MSG_APPL_REQUEST			3<<1
#define AUTH_MSG_APPL_REQUEST_MUTUAL		4<<1
#define AUTH_MSG_ERR_REPLY			5<<1
#define AUTH_MSG_PRIVATE			6<<1
#define AUTH_MSG_SAFE				7<<1
#define AUTH_MSG_APPL_ERR			8<<1
#define AUTH_MSG_DIE				63<<1

#define KERB_ERR_OK				0
#define KERB_ERR_NAME_EXP			1
#define KERB_ERR_SERVICE_EXP			2
#define KERB_ERR_AUTH_EXP			3
#define KERB_ERR_PKT_VER			4
#define KERB_ERR_NAME_MAST_KEY_VER		5
#define KERB_ERR_SERV_MAST_KEY_VER		6
#define KERB_ERR_BYTE_ORDER			7
#define KERB_ERR_PRINCIPAL_UNKNOWN		8
#define KERB_ERR_PRINCIPAL_NOT_UNIQUE		9
#define KERB_ERR_NULL_KEY			10

struct krb {
	nd_uint8_t pvno;	/* Protocol Version */
	nd_uint8_t type;	/* Type+B */
};

static const struct tok type2str[] = {
	{ AUTH_MSG_KDC_REQUEST,		"KDC_REQUEST" },
	{ AUTH_MSG_KDC_REPLY,		"KDC_REPLY" },
	{ AUTH_MSG_APPL_REQUEST,	"APPL_REQUEST" },
	{ AUTH_MSG_APPL_REQUEST_MUTUAL,	"APPL_REQUEST_MUTUAL" },
	{ AUTH_MSG_ERR_REPLY,		"ERR_REPLY" },
	{ AUTH_MSG_PRIVATE,		"PRIVATE" },
	{ AUTH_MSG_SAFE,		"SAFE" },
	{ AUTH_MSG_APPL_ERR,		"APPL_ERR" },
	{ AUTH_MSG_DIE,			"DIE" },
	{ 0,				NULL }
};

static const struct tok kerr2str[] = {
	{ KERB_ERR_OK,			"OK" },
	{ KERB_ERR_NAME_EXP,		"NAME_EXP" },
	{ KERB_ERR_SERVICE_EXP,		"SERVICE_EXP" },
	{ KERB_ERR_AUTH_EXP,		"AUTH_EXP" },
	{ KERB_ERR_PKT_VER,		"PKT_VER" },
	{ KERB_ERR_NAME_MAST_KEY_VER,	"NAME_MAST_KEY_VER" },
	{ KERB_ERR_SERV_MAST_KEY_VER,	"SERV_MAST_KEY_VER" },
	{ KERB_ERR_BYTE_ORDER,		"BYTE_ORDER" },
	{ KERB_ERR_PRINCIPAL_UNKNOWN,	"PRINCIPAL_UNKNOWN" },
	{ KERB_ERR_PRINCIPAL_NOT_UNIQUE,"PRINCIPAL_NOT_UNIQUE" },
	{ KERB_ERR_NULL_KEY,		"NULL_KEY"},
	{ 0,				NULL}
};

static const u_char *
c_print(netdissect_options *ndo,
        const u_char *s, const u_char *ep)
{
	u_char c;
	int flag;

	flag = 1;
	while (s < ep) {
		c = GET_U_1(s);
		s++;
		if (c == '\0') {
			flag = 0;
			break;
		}
		fn_print_char(ndo, c);
	}
	if (flag)
		return NULL;
	return (s);
}

static const u_char *
krb4_print_hdr(netdissect_options *ndo,
               const u_char *cp)
{
	cp += 2;

#define PRINT		if ((cp = c_print(ndo, cp, ndo->ndo_snapend)) == NULL) goto trunc

	PRINT;
	ND_PRINT(".");
	PRINT;
	ND_PRINT("@");
	PRINT;
	return (cp);

trunc:
	nd_print_trunc(ndo);
	return (NULL);

#undef PRINT
}

static void
krb4_print(netdissect_options *ndo,
           const u_char *cp)
{
	const struct krb *kp;
	u_char type;
	u_short len;

#define PRINT		if ((cp = c_print(ndo, cp, ndo->ndo_snapend)) == NULL) goto trunc
/*  True if struct krb is little endian */
#define IS_LENDIAN(kp)	((GET_U_1((kp)->type) & 0x01) != 0)
#define KTOHSP(kp, cp)	(IS_LENDIAN(kp) ? GET_LE_U_2(cp) : GET_BE_U_2(cp))

	kp = (const struct krb *)cp;

	type = GET_U_1(kp->type) & (0xFF << 1);

	ND_PRINT(" %s %s: ",
	    IS_LENDIAN(kp) ? "le" : "be", tok2str(type2str, NULL, type));

	switch (type) {

	case AUTH_MSG_KDC_REQUEST:
		if ((cp = krb4_print_hdr(ndo, cp)) == NULL)
			return;
		cp += 4;	/* ctime */
		ND_PRINT(" %umin ", GET_U_1(cp) * 5);
		cp++;
		PRINT;
		ND_PRINT(".");
		PRINT;
		break;

	case AUTH_MSG_APPL_REQUEST:
		cp += 2;
		ND_PRINT("v%u ", GET_U_1(cp));
		cp++;
		PRINT;
		ND_PRINT(" (%u)", GET_U_1(cp));
		cp++;
		ND_PRINT(" (%u)", GET_U_1(cp));
		break;

	case AUTH_MSG_KDC_REPLY:
		if ((cp = krb4_print_hdr(ndo, cp)) == NULL)
			return;
		cp += 10;	/* timestamp + n + exp + kvno */
		len = KTOHSP(kp, cp);
		ND_PRINT(" (%u)", len);
		break;

	case AUTH_MSG_ERR_REPLY:
		if ((cp = krb4_print_hdr(ndo, cp)) == NULL)
			return;
		cp += 4; 	  /* timestamp */
		ND_PRINT(" %s ", tok2str(kerr2str, NULL, KTOHSP(kp, cp)));
		cp += 4;
		PRINT;
		break;

	default:
		ND_PRINT("(unknown)");
		break;
	}

	return;
trunc:
	nd_print_trunc(ndo);
}

void
krb_print(netdissect_options *ndo,
          const u_char *dat)
{
	const struct krb *kp;

	ndo->ndo_protocol = "krb";
	kp = (const struct krb *)dat;

	if (dat >= ndo->ndo_snapend) {
		nd_print_trunc(ndo);
		return;
	}

	switch (GET_U_1(kp->pvno)) {

	case 1:
	case 2:
	case 3:
		ND_PRINT(" v%u", GET_U_1(kp->pvno));
		break;

	case 4:
		ND_PRINT(" v%u", GET_U_1(kp->pvno));
		krb4_print(ndo, (const u_char *)kp);
		break;

	case 106:
	case 107:
		ND_PRINT(" v5");
		/* Decode ASN.1 here "someday" */
		break;
	}
}
