/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
/* \summary: Extensible Authentication Protocol (EAP) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

#define	EAP_FRAME_TYPE_PACKET		0
#define	EAP_FRAME_TYPE_START		1
#define	EAP_FRAME_TYPE_LOGOFF		2
#define	EAP_FRAME_TYPE_KEY		3
#define	EAP_FRAME_TYPE_ENCAP_ASF_ALERT	4

struct eap_frame_t {
    nd_uint8_t  version;
    nd_uint8_t  type;
    nd_uint16_t length;
};

static const struct tok eap_frame_type_values[] = {
    { EAP_FRAME_TYPE_PACKET,      	"EAP packet" },
    { EAP_FRAME_TYPE_START,    		"EAPOL start" },
    { EAP_FRAME_TYPE_LOGOFF,      	"EAPOL logoff" },
    { EAP_FRAME_TYPE_KEY,      		"EAPOL key" },
    { EAP_FRAME_TYPE_ENCAP_ASF_ALERT, 	"Encapsulated ASF alert" },
    { 0, NULL}
};

/* RFC 3748 */
struct eap_packet_t {
    nd_uint8_t  code;
    nd_uint8_t  id;
    nd_uint16_t length;
};

#define		EAP_REQUEST	1
#define		EAP_RESPONSE	2
#define		EAP_SUCCESS	3
#define		EAP_FAILURE	4

static const struct tok eap_code_values[] = {
    { EAP_REQUEST,	"Request" },
    { EAP_RESPONSE,	"Response" },
    { EAP_SUCCESS,	"Success" },
    { EAP_FAILURE,	"Failure" },
    { 0, NULL}
};

#define		EAP_TYPE_NO_PROPOSED	0
#define		EAP_TYPE_IDENTITY	1
#define		EAP_TYPE_NOTIFICATION	2
#define		EAP_TYPE_NAK		3
#define		EAP_TYPE_MD5_CHALLENGE	4
#define		EAP_TYPE_OTP		5
#define		EAP_TYPE_GTC		6
#define		EAP_TYPE_TLS		13		/* RFC 2716 */
#define		EAP_TYPE_SIM		18		/* RFC 4186 */
#define		EAP_TYPE_TTLS		21		/* draft-funk-eap-ttls-v0-01.txt */
#define		EAP_TYPE_AKA		23		/* RFC 4187 */
#define		EAP_TYPE_FAST		43		/* RFC 4851 */
#define		EAP_TYPE_EXPANDED_TYPES	254
#define		EAP_TYPE_EXPERIMENTAL	255

static const struct tok eap_type_values[] = {
    { EAP_TYPE_NO_PROPOSED,	"No proposed" },
    { EAP_TYPE_IDENTITY,	"Identity" },
    { EAP_TYPE_NOTIFICATION,    "Notification" },
    { EAP_TYPE_NAK,      	"Nak" },
    { EAP_TYPE_MD5_CHALLENGE,   "MD5-challenge" },
    { EAP_TYPE_OTP,      	"OTP" },
    { EAP_TYPE_GTC,      	"GTC" },
    { EAP_TYPE_TLS,      	"TLS" },
    { EAP_TYPE_SIM,      	"SIM" },
    { EAP_TYPE_TTLS,      	"TTLS" },
    { EAP_TYPE_AKA,      	"AKA" },
    { EAP_TYPE_FAST,      	"FAST" },
    { EAP_TYPE_EXPANDED_TYPES,  "Expanded types" },
    { EAP_TYPE_EXPERIMENTAL,    "Experimental" },
    { 0, NULL}
};

#define EAP_TLS_EXTRACT_BIT_L(x) 	(((x)&0x80)>>7)

/* RFC 2716 - EAP TLS bits */
#define EAP_TLS_FLAGS_LEN_INCLUDED		(1 << 7)
#define EAP_TLS_FLAGS_MORE_FRAGMENTS		(1 << 6)
#define EAP_TLS_FLAGS_START			(1 << 5)

static const struct tok eap_tls_flags_values[] = {
	{ EAP_TLS_FLAGS_LEN_INCLUDED, "L bit" },
	{ EAP_TLS_FLAGS_MORE_FRAGMENTS, "More fragments bit"},
	{ EAP_TLS_FLAGS_START, "Start bit"},
	{ 0, NULL}
};

#define EAP_TTLS_VERSION(x)		((x)&0x07)

/* EAP-AKA and EAP-SIM - RFC 4187 */
#define EAP_AKA_CHALLENGE		1
#define EAP_AKA_AUTH_REJECT		2
#define EAP_AKA_SYNC_FAILURE		4
#define EAP_AKA_IDENTITY		5
#define EAP_SIM_START			10
#define EAP_SIM_CHALLENGE		11
#define EAP_AKA_NOTIFICATION		12
#define EAP_AKA_REAUTH			13
#define EAP_AKA_CLIENT_ERROR		14

static const struct tok eap_aka_subtype_values[] = {
    { EAP_AKA_CHALLENGE,	"Challenge" },
    { EAP_AKA_AUTH_REJECT,	"Auth reject" },
    { EAP_AKA_SYNC_FAILURE,	"Sync failure" },
    { EAP_AKA_IDENTITY,		"Identity" },
    { EAP_SIM_START,		"Start" },
    { EAP_SIM_CHALLENGE,	"Challenge" },
    { EAP_AKA_NOTIFICATION,	"Notification" },
    { EAP_AKA_REAUTH,		"Reauth" },
    { EAP_AKA_CLIENT_ERROR,	"Client error" },
    { 0, NULL}
};

/*
 * Print EAP requests / responses
 */
void
eap_print(netdissect_options *ndo,
          const u_char *cp,
          u_int length)
{
    u_int type, subtype, len;
    int count;

    type = GET_U_1(cp);
    len = GET_BE_U_2(cp + 2);
    if(len != length) {
       goto trunc;
    }
    ND_PRINT("%s (%u), id %u, len %u",
            tok2str(eap_code_values, "unknown", type),
            type,
            GET_U_1((cp + 1)),
            len);

    ND_TCHECK_LEN(cp, len);

    if (type == EAP_REQUEST || type == EAP_RESPONSE) {
        /* RFC 3748 Section 4.1 */
        subtype = GET_U_1(cp + 4);
        ND_PRINT("\n\t\t Type %s (%u)",
                tok2str(eap_type_values, "unknown", subtype),
                subtype);

        switch (subtype) {
            case EAP_TYPE_IDENTITY:
                if (len - 5 > 0) {
                    ND_PRINT(", Identity: ");
                    nd_printjnp(ndo, cp + 5, len - 5);
                }
                break;

            case EAP_TYPE_NOTIFICATION:
                if (len - 5 > 0) {
                    ND_PRINT(", Notification: ");
                    nd_printjnp(ndo, cp + 5, len - 5);
                }
                break;

            case EAP_TYPE_NAK:
                count = 5;

                /*
                 * one or more octets indicating
                 * the desired authentication
                 * type one octet per type
                 */
                while (count < (int)len) {
                    ND_PRINT(" %s (%u),",
                           tok2str(eap_type_values, "unknown", GET_U_1((cp + count))),
                           GET_U_1(cp + count));
                    count++;
                }
                break;

            case EAP_TYPE_TTLS:
            case EAP_TYPE_TLS:
                if (subtype == EAP_TYPE_TTLS)
                    ND_PRINT(" TTLSv%u",
                           EAP_TTLS_VERSION(GET_U_1((cp + 5))));
                ND_PRINT(" flags [%s] 0x%02x,",
                       bittok2str(eap_tls_flags_values, "none", GET_U_1((cp + 5))),
                       GET_U_1(cp + 5));

                if (EAP_TLS_EXTRACT_BIT_L(GET_U_1(cp + 5))) {
                    ND_PRINT(" len %u", GET_BE_U_4(cp + 6));
                }
                break;

            case EAP_TYPE_FAST:
                ND_PRINT(" FASTv%u",
                       EAP_TTLS_VERSION(GET_U_1((cp + 5))));
                ND_PRINT(" flags [%s] 0x%02x,",
                       bittok2str(eap_tls_flags_values, "none", GET_U_1((cp + 5))),
                       GET_U_1(cp + 5));

                if (EAP_TLS_EXTRACT_BIT_L(GET_U_1(cp + 5))) {
                    ND_PRINT(" len %u", GET_BE_U_4(cp + 6));
                }

                /* FIXME - TLV attributes follow */
                break;

            case EAP_TYPE_AKA:
            case EAP_TYPE_SIM:
                ND_PRINT(" subtype [%s] 0x%02x,",
                       tok2str(eap_aka_subtype_values, "unknown", GET_U_1((cp + 5))),
                       GET_U_1(cp + 5));

                /* FIXME - TLV attributes follow */
                break;

            case EAP_TYPE_MD5_CHALLENGE:
            case EAP_TYPE_OTP:
            case EAP_TYPE_GTC:
            case EAP_TYPE_EXPANDED_TYPES:
            case EAP_TYPE_EXPERIMENTAL:
            default:
                break;
        }
    }
    return;
trunc:
    nd_print_trunc(ndo);
}

void
eapol_print(netdissect_options *ndo,
            const u_char *cp)
{
    const struct eap_frame_t *eap;
    u_int eap_type, eap_len;

    ndo->ndo_protocol = "eap";
    eap = (const struct eap_frame_t *)cp;
    ND_TCHECK_SIZE(eap);
    eap_type = GET_U_1(eap->type);

    ND_PRINT("%s (%u) v%u, len %u",
           tok2str(eap_frame_type_values, "unknown", eap_type),
           eap_type,
           GET_U_1(eap->version),
           GET_BE_U_2(eap->length));
    if (ndo->ndo_vflag < 1)
        return;

    cp += sizeof(struct eap_frame_t);
    eap_len = GET_BE_U_2(eap->length);

    switch (eap_type) {
    case EAP_FRAME_TYPE_PACKET:
        if (eap_len == 0)
            goto trunc;
        ND_PRINT(", ");
        eap_print(ndo, cp, eap_len);
        return;
    case EAP_FRAME_TYPE_LOGOFF:
    case EAP_FRAME_TYPE_ENCAP_ASF_ALERT:
    default:
        break;
    }
    return;

 trunc:
    nd_print_trunc(ndo);
}
