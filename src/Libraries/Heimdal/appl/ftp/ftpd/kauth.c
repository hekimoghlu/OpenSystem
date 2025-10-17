/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "ftpd_locl.h"

RCSID("$Id$");

#if defined(KRB5)

int do_destroy_tickets = 1;
char *k5ccname;

#endif

#ifdef KRB5

static void
dest_cc(void)
{
    krb5_context context;
    krb5_error_code ret;
    krb5_ccache id;

    ret = krb5_init_context(&context);
    if (ret == 0) {
	if (k5ccname)
	    ret = krb5_cc_resolve(context, k5ccname, &id);
	else
	    ret = krb5_cc_default (context, &id);
	if (ret)
	    krb5_free_context(context);
    }
    if (ret == 0) {
	krb5_cc_destroy(context, id);
	krb5_free_context (context);
    }
}
#endif

#if defined(KRB5)

/*
 * Only destroy if we created the tickets
 */

void
cond_kdestroy(void)
{
    if (do_destroy_tickets) {
#if KRB5
	dest_cc();
#endif
	do_destroy_tickets = 0;
    }
    afsunlog();
}

void
kdestroy(void)
{
#if KRB5
    dest_cc();
#endif
    afsunlog();
    reply(200, "Tickets destroyed");
}


void
afslog(const char *cell, int quiet)
{
    if(k_hasafs()) {
#ifdef KRB5
	krb5_context context;
	krb5_error_code ret;
	krb5_ccache id;

	ret = krb5_init_context(&context);
	if (ret == 0) {
	    if (k5ccname)
		ret = krb5_cc_resolve(context, k5ccname, &id);
	    else
		ret = krb5_cc_default(context, &id);
	    if (ret)
		krb5_free_context(context);
	}
	if (ret == 0) {
	    krb5_afslog(context, id, cell, 0);
	    krb5_cc_close (context, id);
	    krb5_free_context (context);
	}
#endif
	if (!quiet)
	    reply(200, "afslog done");
    } else {
	if (!quiet)
	    reply(200, "no AFS present");
    }
}

void
afsunlog(void)
{
    if(k_hasafs())
	k_unlog();
}

#else
int ftpd_afslog_placeholder;
#endif /* KRB5 */
