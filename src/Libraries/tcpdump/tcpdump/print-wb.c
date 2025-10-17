/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
/* \summary: White Board printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "addrtoname.h"
#include "extract.h"


#if 0
/*
 * Largest packet size.  Everything should fit within this space.
 * For instance, multiline objects are sent piecewise.
 */
#define MAXFRAMESIZE 1024
#endif

/*
 * Multiple drawing ops can be sent in one packet.  Each one starts on a
 * an even multiple of DOP_ALIGN bytes, which must be a power of two.
 */
#define DOP_ALIGN 4
#define DOP_ROUNDUP(x)	roundup2(x, DOP_ALIGN)
#define DOP_NEXT(d)\
	((const struct dophdr *)((const u_char *)(d) + \
				DOP_ROUNDUP(GET_BE_U_2((d)->dh_len) + sizeof(*(d)))))

/*
 * Format of the whiteboard packet header.
 * The transport level header.
 */
struct pkt_hdr {
	nd_uint32_t ph_src;	/* site id of source */
	nd_uint32_t ph_ts;	/* time stamp (for skew computation) */
	nd_uint16_t ph_version;	/* version number */
	nd_uint8_t ph_type;	/* message type */
	nd_uint8_t ph_flags;	/* message flags */
};

/* Packet types */
#define PT_DRAWOP	0	/* drawing operation */
#define PT_ID		1	/* announcement packet */
#define PT_RREQ		2	/* repair request */
#define PT_RREP		3	/* repair reply */
#define PT_KILL		4	/* terminate participation */
#define PT_PREQ         5       /* page vector request */
#define PT_PREP         7       /* page vector reply */

#if 0
#ifdef PF_USER
#undef PF_USER			/* {Digital,Tru64} UNIX define this, alas */
#endif

/* flags */
#define PF_USER		0x01	/* hint that packet has interactive data */
#define PF_VIS		0x02	/* only visible ops wanted */
#endif

struct PageID {
	nd_uint32_t p_sid;		/* session id of initiator */
	nd_uint32_t p_uid;		/* page number */
};

struct dophdr {
	nd_uint32_t	dh_ts;		/* sender's timestamp */
	nd_uint16_t	dh_len;		/* body length */
	nd_uint8_t	dh_flags;
	nd_uint8_t	dh_type;	/* body type */
	/* body follows */
};
/*
 * Drawing op sub-types.
 */
#define DT_RECT         2
#define DT_LINE         3
#define DT_ML           4
#define DT_DEL          5
#define DT_XFORM        6
#define DT_ELL          7
#define DT_CHAR         8
#define DT_STR          9
#define DT_NOP          10
#define DT_PSCODE       11
#define DT_PSCOMP       12
#define DT_REF          13
#define DT_SKIP         14
#define DT_HOLE         15
static const struct tok dop_str[] = {
	{ DT_RECT,   "RECT"   },
	{ DT_LINE,   "LINE"   },
	{ DT_ML,     "ML"     },
	{ DT_DEL,    "DEL"    },
	{ DT_XFORM,  "XFORM"  },
	{ DT_ELL,    "ELL"    },
	{ DT_CHAR,   "CHAR"   },
	{ DT_STR,    "STR"    },
	{ DT_NOP,    "NOP"    },
	{ DT_PSCODE, "PSCODE" },
	{ DT_PSCOMP, "PSCOMP" },
	{ DT_REF,    "REF"    },
	{ DT_SKIP,   "SKIP"   },
	{ DT_HOLE,   "HOLE"   },
	{ 0, NULL }
};

/*
 * A drawing operation.
 */
struct pkt_dop {
	struct PageID pd_page;	/* page that operations apply to */
	nd_uint32_t	pd_sseq;	/* start sequence number */
	nd_uint32_t	pd_eseq;	/* end sequence number */
	/* drawing ops follow */
};

/*
 * A repair request.
 */
struct pkt_rreq {
        nd_uint32_t pr_id;        /* source id of drawops to be repaired */
        struct PageID pr_page;    /* page of drawops */
        nd_uint32_t pr_sseq;      /* start seqno */
        nd_uint32_t pr_eseq;      /* end seqno */
};

/*
 * A repair reply.
 */
struct pkt_rrep {
	nd_uint32_t pr_id;	/* original site id of ops  */
	struct pkt_dop pr_dop;
	/* drawing ops follow */
};

struct id_off {
        nd_uint32_t id;
        nd_uint32_t off;
};

struct pgstate {
	nd_uint32_t slot;
	struct PageID page;
	nd_uint16_t nid;
	nd_uint16_t rsvd;
        /* seqptr's */
};

/*
 * An announcement packet.
 */
struct pkt_id {
	nd_uint32_t pi_mslot;
        struct PageID    pi_mpage;        /* current page */
	struct pgstate pi_ps;
        /* seqptr's */
        /* null-terminated site name */
};

struct pkt_preq {
        struct PageID  pp_page;
        nd_uint32_t  pp_low;
        nd_uint32_t  pp_high;
};

struct pkt_prep {
        nd_uint32_t  pp_n;           /* size of pageid array */
        /* pgstate's follow */
};

static int
wb_id(netdissect_options *ndo,
      const struct pkt_id *id, u_int len)
{
	u_int i;
	const u_char *sitename;
	const struct id_off *io;
	char c;
	u_int nid;

	ND_PRINT(" wb-id:");
	if (len < sizeof(*id))
		return (-1);
	len -= sizeof(*id);

	ND_PRINT(" %u/%s:%u (max %u/%s:%u) ",
	       GET_BE_U_4(id->pi_ps.slot),
	       GET_IPADDR_STRING(id->pi_ps.page.p_sid),
	       GET_BE_U_4(id->pi_ps.page.p_uid),
	       GET_BE_U_4(id->pi_mslot),
	       GET_IPADDR_STRING(id->pi_mpage.p_sid),
	       GET_BE_U_4(id->pi_mpage.p_uid));
	/* now the rest of the fixed-size part of struct pkt_id */
	ND_TCHECK_SIZE(id);

	nid = GET_BE_U_2(id->pi_ps.nid);
	if (len < sizeof(*io) * nid)
		return (-1);
	len -= sizeof(*io) * nid;
	io = (const struct id_off *)(id + 1);
	sitename = (const u_char *)(io + nid);

	c = '<';
	for (i = 0; i < nid; ++io, ++i) {
		ND_PRINT("%c%s:%u",
		    c, GET_IPADDR_STRING(io->id), GET_BE_U_4(io->off));
		c = ',';
	}
	ND_PRINT("> \"");
	nd_printjnp(ndo, sitename, len);
	ND_PRINT("\"");
	return (0);
}

static int
wb_rreq(netdissect_options *ndo,
        const struct pkt_rreq *rreq, u_int len)
{
	ND_PRINT(" wb-rreq:");
	if (len < sizeof(*rreq))
		return (-1);

	ND_PRINT(" please repair %s %s:%u<%u:%u>",
	       GET_IPADDR_STRING(rreq->pr_id),
	       GET_IPADDR_STRING(rreq->pr_page.p_sid),
	       GET_BE_U_4(rreq->pr_page.p_uid),
	       GET_BE_U_4(rreq->pr_sseq),
	       GET_BE_U_4(rreq->pr_eseq));
	return (0);
}

static int
wb_preq(netdissect_options *ndo,
        const struct pkt_preq *preq, u_int len)
{
	ND_PRINT(" wb-preq:");
	if (len < sizeof(*preq))
		return (-1);

	ND_PRINT(" need %u/%s:%u",
	       GET_BE_U_4(preq->pp_low),
	       GET_IPADDR_STRING(preq->pp_page.p_sid),
	       GET_BE_U_4(preq->pp_page.p_uid));
	/* now the rest of the fixed-size part of struct pkt_req */
	ND_TCHECK_SIZE(preq);
	return (0);
}

static int
wb_prep(netdissect_options *ndo,
        const struct pkt_prep *prep, u_int len)
{
	u_int n;
	const struct pgstate *ps;

	ND_PRINT(" wb-prep:");
	if (len < sizeof(*prep))
		return (-1);
	n = GET_BE_U_4(prep->pp_n);
	ps = (const struct pgstate *)(prep + 1);
	while (n != 0) {
		const struct id_off *io, *ie;
		char c = '<';

		ND_PRINT(" %u/%s:%u",
		    GET_BE_U_4(ps->slot),
		    GET_IPADDR_STRING(ps->page.p_sid),
		    GET_BE_U_4(ps->page.p_uid));
		/* now the rest of the fixed-size part of struct pgstate */
		ND_TCHECK_SIZE(ps);
		io = (const struct id_off *)(ps + 1);
		for (ie = io + GET_U_1(ps->nid); io < ie; ++io) {
			ND_PRINT("%c%s:%u", c, GET_IPADDR_STRING(io->id),
			    GET_BE_U_4(io->off));
			c = ',';
		}
		ND_PRINT(">");
		ps = (const struct pgstate *)io;
		n--;
	}
	return 0;
}

static void
wb_dops(netdissect_options *ndo, const struct pkt_dop *dop,
        uint32_t ss, uint32_t es)
{
	const struct dophdr *dh = (const struct dophdr *)((const u_char *)dop + sizeof(*dop));

	ND_PRINT(" <");
	for ( ; ss <= es; ++ss) {
		u_int t;

		t = GET_U_1(dh->dh_type);

		ND_PRINT(" %s", tok2str(dop_str, "dop-%u!", t));
		if (t == DT_SKIP || t == DT_HOLE) {
			uint32_t ts = GET_BE_U_4(dh->dh_ts);
			ND_PRINT("%u", ts - ss + 1);
			if (ss > ts || ts > es) {
				ND_PRINT("[|]");
				if (ts < ss)
					return;
			}
			ss = ts;
		}
		dh = DOP_NEXT(dh);
	}
	ND_PRINT(" >");
}

static int
wb_rrep(netdissect_options *ndo,
        const struct pkt_rrep *rrep, u_int len)
{
	const struct pkt_dop *dop = &rrep->pr_dop;

	ND_PRINT(" wb-rrep:");
	if (len < sizeof(*rrep))
		return (-1);
	len -= sizeof(*rrep);

	ND_PRINT(" for %s %s:%u<%u:%u>",
	    GET_IPADDR_STRING(rrep->pr_id),
	    GET_IPADDR_STRING(dop->pd_page.p_sid),
	    GET_BE_U_4(dop->pd_page.p_uid),
	    GET_BE_U_4(dop->pd_sseq),
	    GET_BE_U_4(dop->pd_eseq));

	if (ndo->ndo_vflag)
		wb_dops(ndo, dop,
		        GET_BE_U_4(dop->pd_sseq),
		        GET_BE_U_4(dop->pd_eseq));
	return (0);
}

static int
wb_drawop(netdissect_options *ndo,
          const struct pkt_dop *dop, u_int len)
{
	ND_PRINT(" wb-dop:");
	if (len < sizeof(*dop))
		return (-1);
	len -= sizeof(*dop);

	ND_PRINT(" %s:%u<%u:%u>",
	    GET_IPADDR_STRING(dop->pd_page.p_sid),
	    GET_BE_U_4(dop->pd_page.p_uid),
	    GET_BE_U_4(dop->pd_sseq),
	    GET_BE_U_4(dop->pd_eseq));

	if (ndo->ndo_vflag)
		wb_dops(ndo, dop,
		        GET_BE_U_4(dop->pd_sseq),
		        GET_BE_U_4(dop->pd_eseq));
	return (0);
}

/*
 * Print whiteboard multicast packets.
 */
void
wb_print(netdissect_options *ndo,
         const u_char *hdr, u_int len)
{
	const struct pkt_hdr *ph;
	uint8_t type;
	int print_result;

	ndo->ndo_protocol = "wb";
	ph = (const struct pkt_hdr *)hdr;
	if (len < sizeof(*ph))
		goto invalid;
	ND_TCHECK_SIZE(ph);
	len -= sizeof(*ph);

	if (GET_U_1(ph->ph_flags))
		ND_PRINT("*");
	type = GET_U_1(ph->ph_type);
	switch (type) {

	case PT_KILL:
		ND_PRINT(" wb-kill");
		return;

	case PT_ID:
		print_result = wb_id(ndo, (const struct pkt_id *)(ph + 1), len);
		break;

	case PT_RREQ:
		print_result = wb_rreq(ndo, (const struct pkt_rreq *)(ph + 1), len);
		break;

	case PT_RREP:
		print_result = wb_rrep(ndo, (const struct pkt_rrep *)(ph + 1), len);
		break;

	case PT_DRAWOP:
		print_result = wb_drawop(ndo, (const struct pkt_dop *)(ph + 1), len);
		break;

	case PT_PREQ:
		print_result = wb_preq(ndo, (const struct pkt_preq *)(ph + 1), len);
		break;

	case PT_PREP:
		print_result = wb_prep(ndo, (const struct pkt_prep *)(ph + 1), len);
		break;

	default:
		ND_PRINT(" wb-%u!", type);
		print_result = -1;
	}
	if (print_result < 0)
		goto invalid;
	return;

invalid:
	nd_print_invalid(ndo);
}
