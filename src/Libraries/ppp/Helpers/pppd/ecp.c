/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
/*
 * ecp.c - PPP Encryption Control Protocol.
 *
 * Copyright (c) 2002 Google, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The name(s) of the authors of this software must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission.
 *
 * THE AUTHORS OF THIS SOFTWARE DISCLAIM ALL WARRANTIES WITH REGARD TO
 * THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS, IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Derived from ccp.c, which is:
 *
 * Copyright (c) 1994-2002 Paul Mackerras. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The name(s) of the authors of this software must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission.
 *
 * 4. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by Paul Mackerras
 *     <paulus@samba.org>".
 *
 * THE AUTHORS OF THIS SOFTWARE DISCLAIM ALL WARRANTIES WITH REGARD TO
 * THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS, IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#define RCSID	"$Id: ecp.c,v 1.6 2005/12/13 06:30:15 lindak Exp $"

#ifndef lint
static const char rcsid[] = RCSID;
#endif

#include <string.h>

#include "pppd.h"
#include "fsm.h"
#include "ecp.h"

static option_t ecp_option_list[] = {
    { "noecp", o_bool, &ecp_protent.enabled_flag,
      "Disable ECP negotiation" },
    { "-ecp", o_bool, &ecp_protent.enabled_flag,
      "Disable ECP negotiation", OPT_ALIAS },

    { NULL }
};

/*
 * Protocol entry points from main code.
 */
static void ecp_init __P((int unit));
/*
static void ecp_open __P((int unit));
static void ecp_close __P((int unit, char *));
static void ecp_lowerup __P((int unit));
static void ecp_lowerdown __P((int));
static void ecp_input __P((int unit, u_char *pkt, int len));
static void ecp_protrej __P((int unit));
*/
static int  ecp_printpkt __P((u_char *pkt, int len,
			      void (*printer) __P((void *, char *, ...)),
			      void *arg));
/*
static void ecp_datainput __P((int unit, u_char *pkt, int len));
*/

struct protent ecp_protent = {
    PPP_ECP,
    ecp_init,
    NULL, /* ecp_input, */
    NULL, /* ecp_protrej, */
    NULL, /* ecp_lowerup, */
    NULL, /* ecp_lowerdown, */
    NULL, /* ecp_open, */
    NULL, /* ecp_close, */
    ecp_printpkt,
    NULL, /* ecp_datainput, */
    0,
    "ECP",
    "Encrypted",
    ecp_option_list,
    NULL,
    NULL,
    NULL,
#ifdef __APPLE__
    NULL,
    NULL,
    NULL,
    NULL
#endif
};

fsm ecp_fsm[NUM_PPP];
ecp_options ecp_wantoptions[NUM_PPP];	/* what to request the peer to use */
ecp_options ecp_gotoptions[NUM_PPP];	/* what the peer agreed to do */
ecp_options ecp_allowoptions[NUM_PPP];	/* what we'll agree to do */
ecp_options ecp_hisoptions[NUM_PPP];	/* what we agreed to do */

static fsm_callbacks ecp_callbacks = {
    NULL, /* ecp_resetci, */
    NULL, /* ecp_cilen, */
    NULL, /* ecp_addci, */
    NULL, /* ecp_ackci, */
    NULL, /* ecp_nakci, */
    NULL, /* ecp_rejci, */
    NULL, /* ecp_reqci, */
    NULL, /* ecp_up, */
    NULL, /* ecp_down, */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* ecp_extcode, */
    "ECP"
};

/*
 * ecp_init - initialize ECP.
 */
static void
ecp_init(unit)
    int unit;
{
    fsm *f = &ecp_fsm[unit];

    f->unit = unit;
    f->protocol = PPP_ECP;
    f->callbacks = &ecp_callbacks;
    fsm_init(f);

    memset(&ecp_wantoptions[unit],  0, sizeof(ecp_options));
    memset(&ecp_gotoptions[unit],   0, sizeof(ecp_options));
    memset(&ecp_allowoptions[unit], 0, sizeof(ecp_options));
    memset(&ecp_hisoptions[unit],   0, sizeof(ecp_options));

}


static int
ecp_printpkt(p, plen, printer, arg)
    u_char *p;
    int plen;
    void (*printer) __P((void *, char *, ...));
    void *arg;
{
    return 0;
}

