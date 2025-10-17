/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "regguts.h"

/*
 * Unknown-error explanation.
 */

static char unk[] = "*** unknown regex error code 0x%x ***";

/*
 * Struct to map among codes, code names, and explanations.
 */

static struct rerr {
    int code;
    const char *name;
    const char *explain;
} rerrs[] = {
    /* The actual table is built from regex.h */
#include "regerrs.h"
    { -1, "", "oops" },		/* explanation special-cased in code */
};

/*
 - regerror - the interface to error numbers
 */
/* ARGSUSED */
size_t				/* Actual space needed (including NUL) */
regerror(
    int code,			/* Error code, or REG_ATOI or REG_ITOA */
    const regex_t *preg,	/* Associated regex_t (unused at present) */
    char *errbuf,		/* Result buffer (unless errbuf_size==0) */
    size_t errbuf_size)		/* Available space in errbuf, can be 0 */
{
    struct rerr *r;
    const char *msg;
    char convbuf[sizeof(unk)+50]; /* 50 = plenty for int */
    size_t len;
    int icode;

    switch (code) {
    case REG_ATOI:		/* Convert name to number */
	for (r = rerrs; r->code >= 0; r++) {
	    if (strcmp(r->name, errbuf) == 0) {
		break;
	    }
	}
	sprintf(convbuf, "%d", r->code); /* -1 for unknown */
	msg = convbuf;
	break;
    case REG_ITOA:		/* Convert number to name */
	icode = atoi(errbuf);	/* Not our problem if this fails */
	for (r = rerrs; r->code >= 0; r++) {
	    if (r->code == icode) {
		break;
	    }
	}
	if (r->code >= 0) {
	    msg = r->name;
	} else {		/* Unknown; tell him the number */
	    sprintf(convbuf, "REG_%u", (unsigned)icode);
	    msg = convbuf;
	}
	break;
    default:			/* A real, normal error code */
	for (r = rerrs; r->code >= 0; r++) {
	    if (r->code == code) {
		break;
	    }
	}
	if (r->code >= 0) {
	    msg = r->explain;
	} else {		/* Unknown; say so */
	    sprintf(convbuf, unk, code);
	    msg = convbuf;
	}
	break;
    }

    len = strlen(msg) + 1;	/* Space needed, including NUL */
    if (errbuf_size > 0) {
	if (errbuf_size > len) {
	    strcpy(errbuf, msg);
	} else {		/* Truncate to fit */
	    strncpy(errbuf, msg, errbuf_size-1);
	    errbuf[errbuf_size-1] = '\0';
	}
    }

    return len;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
