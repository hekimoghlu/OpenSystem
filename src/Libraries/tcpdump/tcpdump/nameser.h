/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#ifndef _NAMESER_H_
#define	_NAMESER_H_

#include <sys/types.h>

/*
 * Define constants based on rfc883
 */
#define PACKETSZ	512		/* maximum packet size */
#define MAXDNAME	256		/* maximum domain name */
#define MAXCDNAME	255		/* maximum compressed domain name */
#define MAXLABEL	63		/* maximum length of domain label */
	/* Number of bytes of fixed size data in query structure */
#define QFIXEDSZ	4
	/* number of bytes of fixed size data in resource record */
#define RRFIXEDSZ	10

/*
 * Currently defined opcodes
 */
#define QUERY		0x0		/* standard query */
#define IQUERY		0x1		/* inverse query */
#define STATUS		0x2		/* nameserver status query */
#if 0
#define xxx		0x3		/* 0x3 reserved */
#endif
	/* non standard - supports ALLOW_UPDATES stuff from Mike Schwartz */
#define UPDATEA		0x9		/* add resource record */
#define UPDATED		0xa		/* delete a specific resource record */
#define UPDATEDA	0xb		/* delete all named resource record */
#define UPDATEM		0xc		/* modify a specific resource record */
#define UPDATEMA	0xd		/* modify all named resource record */

#define ZONEINIT	0xe		/* initial zone transfer */
#define ZONEREF		0xf		/* incremental zone referesh */

/*
 * Undefine various #defines from various System V-flavored OSes (Solaris,
 * SINIX, HP-UX) so the compiler doesn't whine that we redefine them.
 */
#ifdef T_NULL
#undef T_NULL
#endif
#ifdef T_OPT
#undef T_OPT
#endif
#ifdef T_UNSPEC
#undef T_UNSPEC
#endif
#ifdef NOERROR
#undef NOERROR
#endif

/*
 * Currently defined response codes
 */
#define NOERROR		0		/* no error */
#define FORMERR		1		/* format error */
#define SERVFAIL	2		/* server failure */
#define NXDOMAIN	3		/* non existent domain */
#define NOTIMP		4		/* not implemented */
#define REFUSED		5		/* query refused */
	/* non standard */
#define NOCHANGE	0xf		/* update failed to change db */

/*
 * Type values for resources and queries
 */
#define T_A		1		/* host address */
#define T_NS		2		/* authoritative server */
#define T_MD		3		/* mail destination */
#define T_MF		4		/* mail forwarder */
#define T_CNAME		5		/* connonical name */
#define T_SOA		6		/* start of authority zone */
#define T_MB		7		/* mailbox domain name */
#define T_MG		8		/* mail group member */
#define T_MR		9		/* mail rename name */
#define T_NULL		10		/* null resource record */
#define T_WKS		11		/* well known service */
#define T_PTR		12		/* domain name pointer */
#define T_HINFO		13		/* host information */
#define T_MINFO		14		/* mailbox information */
#define T_MX		15		/* mail routing information */
#define T_TXT		16		/* text strings */
#define	T_RP		17		/* responsible person */
#define	T_AFSDB		18		/* AFS cell database */
#define T_X25		19		/* X_25 calling address */
#define T_ISDN		20		/* ISDN calling address */
#define T_RT		21		/* router */
#define	T_NSAP		22		/* NSAP address */
#define	T_NSAP_PTR	23		/* reverse lookup for NSAP */
#define T_SIG		24		/* security signature */
#define T_KEY		25		/* security key */
#define T_PX		26		/* X.400 mail mapping */
#define T_GPOS		27		/* geographical position (withdrawn) */
#define T_AAAA		28		/* IP6 Address */
#define T_LOC		29		/* Location Information */
#define T_NXT		30		/* Next Valid Name in Zone */
#define T_EID		31		/* Endpoint identifier */
#define T_NIMLOC	32		/* Nimrod locator */
#define T_SRV		33		/* Server selection */
#define T_ATMA		34		/* ATM Address */
#define T_NAPTR		35		/* Naming Authority PoinTeR */
#define T_KX		36		/* Key Exchanger */
#define T_CERT		37		/* Certificates in the DNS */
#define T_A6		38		/* IP6 address */
#define T_DNAME		39		/* non-terminal redirection */
#define T_SINK		40		/* unknown */
#define T_OPT		41		/* EDNS0 option (meta-RR) */
#define T_APL		42		/* lists of address prefixes */
#define T_DS		43		/* Delegation Signer */
#define T_SSHFP		44		/* SSH Fingerprint */
#define T_IPSECKEY	45		/* IPsec keying material */
#define T_RRSIG		46		/* new security signature */
#define T_NSEC		47		/* provable insecure information */
#define T_DNSKEY	48		/* new security key */
	/* non standard */
#define T_SPF		99		/* sender policy framework */
#define T_UINFO		100		/* user (finger) information */
#define T_UID		101		/* user ID */
#define T_GID		102		/* group ID */
#define T_UNSPEC	103		/* Unspecified format (binary data) */
#define T_UNSPECA	104		/* "unspecified ASCII". Ugly MIT hack */
	/* Query type values which do not appear in resource records */
#define T_TKEY		249		/* Transaction Key [RFC2930] */
#define T_TSIG		250		/* Transaction Signature [RFC2845] */
#define T_IXFR		251		/* incremental transfer [RFC1995] */
#define T_AXFR		252		/* transfer zone of authority */
#define T_MAILB		253		/* transfer mailbox records */
#define T_MAILA		254		/* transfer mail agent records */
#define T_ANY		255		/* wildcard match */
#define T_URI		256		/* uri records [RFC7553] */

/*
 * Values for class field
 */

#define C_IN		1		/* the arpa internet */
#define C_CHAOS		3		/* for chaos net (MIT) */
#define C_HS		4		/* for Hesiod name server (MIT) (XXX) */
	/* Query class values which do not appear in resource records */
#define C_ANY		255		/* wildcard match */
#define C_QU		0x8000		/* mDNS QU flag in queries */
#define C_CACHE_FLUSH	0x8000		/* mDNS cache flush flag in replies */

/*
 * Values for EDNS option types
 */
#define E_LLQ           1       /* long lived queries protocol */
#define E_UL            2       /* dynamic dns update leases */
#define E_NSID          3       /* name server identifier */
#define E_DAU           5       /* signal DNSSEC algorithm understood */
#define E_DHU           6       /* signal DS hash understood */
#define E_N3U           7       /* signal NSEC3 hash understood */
#define E_ECS           8       /* EDNS client subnet */
#define E_EXPIRE        9       /* zone expiration */
#define E_COOKIE        10      /* DNS cookies */
#define E_KEEPALIVE     11      /* TCP keepalive */
#define E_PADDING       12      /* pad DNS messages */
#define E_CHAIN         13      /* chain DNS queries */
#define E_KEYTAG        14      /* EDNS key tag */
#define E_CLIENTTAG     16      /* EDNS client tag */
#define E_SERVERTAG     17      /* EDNS server tag */

/*
 * Values for DNSSEC Algorithms
 * https://www.iana.org/assignments/dns-sec-alg-numbers/dns-sec-alg-numbers.xhtml
 */

#define A_DELETE                0
#define A_RSAMD5                1
#define A_DH                    2
#define A_DSA                   3
#define A_RSASHA1               5
#define A_DSA_NSEC3_SHA1        6
#define A_RSASHA1_NSEC3_SHA1    7
#define A_RSASHA256             8
#define A_RSASHA512             10
#define A_ECC_GOST              12
#define A_ECDSAP256SHA256       13
#define A_ECDSAP384SHA384       14
#define A_ED25519               15
#define A_ED448                 16
#define A_INDIRECT              252
#define A_PRIVATEDNS            253
#define A_PRIVATEOID            254

/*
 * Values for NSEC3 algorithms
 * https://www.iana.org/assignments/dnssec-nsec3-parameters/dnssec-nsec3-parameters.xhtml
 */
#define NSEC_SHA1   1

/*
 * Values for delegation signer algorithms
 * https://www.iana.org/assignments/ds-rr-types/ds-rr-types.xhtml
 */
#define DS_SHA1     1
#define DS_SHA256   2
#define DS_GOST     3
#define DS_SHA384   4


/*
 * Status return codes for T_UNSPEC conversion routines
 */
#define CONV_SUCCESS 0
#define CONV_OVERFLOW -1
#define CONV_BADFMT -2
#define CONV_BADCKSUM -3
#define CONV_BADBUFLEN -4

/*
 * Structure for query header.
 */
typedef struct {
	nd_uint16_t id;		/* query identification number */
	nd_uint16_t flags;	/* QR, Opcode, AA, TC, RD, RA, RCODE */
	nd_uint16_t qdcount;	/* number of question entries */
	nd_uint16_t ancount;	/* number of answer entries */
	nd_uint16_t nscount;	/* number of authority entries */
	nd_uint16_t arcount;	/* number of resource entries */
} dns_header_t;

/*
 * Macros for subfields of flag fields.
 */
#define DNS_QR(flags)		((flags) & 0x8000)	/* response flag */
#define DNS_OPCODE(flags)	(((flags) >> 11) & 0xF)	/* purpose of message */
#define DNS_AA(flags)		(flags & 0x0400)	/* authoritative answer */
#define DNS_TC(flags)		(flags & 0x0200)	/* truncated message */
#define DNS_RD(flags)		(flags & 0x0100)	/* recursion desired */
#define DNS_RA(flags)		(flags & 0x0080)	/* recursion available */
#define DNS_AD(flags)		(flags & 0x0020)	/* authentic data from named */
#define DNS_CD(flags)		(flags & 0x0010)	/* checking disabled by resolver */
#define DNS_RCODE(flags)	(flags & 0x000F)	/* response code */

/*
 * Defines for handling compressed domain names, EDNS0 labels, etc.
 */
#define TYPE_MASK	0xc0	/* mask for the type bits of the item */
#define TYPE_INDIR	0xc0	/* 11.... - pointer */
#define TYPE_RESERVED	0x80	/* 10.... - reserved */
#define TYPE_EDNS0	0x40	/* 01.... - EDNS(0) label */
#define TYPE_LABEL	0x00	/* 00.... - regular label */
#  define EDNS0_ELT_BITLABEL 0x01

#endif /* !_NAMESER_H_ */
