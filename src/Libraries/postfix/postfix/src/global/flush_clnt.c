/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
/* System library. */

#include "sys_defs.h"
#include <unistd.h>
#include <stdarg.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_flush.h>
#include <mail_params.h>
#include <domain_list.h>
#include <match_parent_style.h>
#include <flush_clnt.h>

/* Application-specific. */

#define STR(x)	vstring_str(x)

static DOMAIN_LIST *flush_domains;

/* flush_init - initialize */

void    flush_init(void)
{
    flush_domains = domain_list_init(VAR_FFLUSH_DOMAINS, MATCH_FLAG_RETURN
				   | match_parent_style(VAR_FFLUSH_DOMAINS),
				     var_fflush_domains);
}

/* flush_purge - house keeping */

int     flush_purge(void)
{
    const char *myname = "flush_purge";
    int     status;

    if (msg_verbose)
	msg_info("%s", myname);

    /*
     * Don't bother the server if the service is turned off.
     */
    if (*var_fflush_domains == 0)
	status = FLUSH_STAT_DENY;
    else
	status = mail_command_client(MAIL_CLASS_PUBLIC, var_flush_service,
			      SEND_ATTR_STR(MAIL_ATTR_REQ, FLUSH_REQ_PURGE),
				     ATTR_TYPE_END);

    if (msg_verbose)
	msg_info("%s: status %d", myname, status);

    return (status);
}

/* flush_refresh - house keeping */

int     flush_refresh(void)
{
    const char *myname = "flush_refresh";
    int     status;

    if (msg_verbose)
	msg_info("%s", myname);

    /*
     * Don't bother the server if the service is turned off.
     */
    if (*var_fflush_domains == 0)
	status = FLUSH_STAT_DENY;
    else
	status = mail_command_client(MAIL_CLASS_PUBLIC, var_flush_service,
			    SEND_ATTR_STR(MAIL_ATTR_REQ, FLUSH_REQ_REFRESH),
				     ATTR_TYPE_END);

    if (msg_verbose)
	msg_info("%s: status %d", myname, status);

    return (status);
}

/* flush_send_site - deliver mail queued for site */

int     flush_send_site(const char *site)
{
    const char *myname = "flush_send_site";
    int     status;

    if (msg_verbose)
	msg_info("%s: site %s", myname, site);

    /*
     * Don't bother the server if the service is turned off, or if the site
     * is not eligible.
     */
    if (flush_domains == 0)
	msg_panic("missing flush client initialization");
    if (domain_list_match(flush_domains, site) != 0) {
	if (warn_compat_break_flush_domains)
	    msg_info("using backwards-compatible default setting "
		     VAR_RELAY_DOMAINS "=$mydestination to flush "
		     "mail for domain \"%s\"", site);
	status = mail_command_client(MAIL_CLASS_PUBLIC, var_flush_service,
			  SEND_ATTR_STR(MAIL_ATTR_REQ, FLUSH_REQ_SEND_SITE),
				     SEND_ATTR_STR(MAIL_ATTR_SITE, site),
				     ATTR_TYPE_END);
    } else if (flush_domains->error == 0)
	status = FLUSH_STAT_DENY;
    else
	status = FLUSH_STAT_FAIL;

    if (msg_verbose)
	msg_info("%s: site %s status %d", myname, site, status);

    return (status);
}

/* flush_send_file - deliver specific message */

int     flush_send_file(const char *queue_id)
{
    const char *myname = "flush_send_file";
    int     status;

    if (msg_verbose)
	msg_info("%s: queue_id %s", myname, queue_id);

    /*
     * Require that the service is turned on.
     */
    status = mail_command_client(MAIL_CLASS_PUBLIC, var_flush_service,
			  SEND_ATTR_STR(MAIL_ATTR_REQ, FLUSH_REQ_SEND_FILE),
				 SEND_ATTR_STR(MAIL_ATTR_QUEUEID, queue_id),
				 ATTR_TYPE_END);

    if (msg_verbose)
	msg_info("%s: queue_id %s status %d", myname, queue_id, status);

    return (status);
}

/* flush_add - inform "fast flush" cache manager */

int     flush_add(const char *site, const char *queue_id)
{
    const char *myname = "flush_add";
    int     status;

    if (msg_verbose)
	msg_info("%s: site %s id %s", myname, site, queue_id);

    /*
     * Don't bother the server if the service is turned off, or if the site
     * is not eligible.
     */
    if (flush_domains == 0)
	msg_panic("missing flush client initialization");
    if (domain_list_match(flush_domains, site) != 0) {
	if (warn_compat_break_flush_domains)
	    msg_info("using backwards-compatible default setting "
		     VAR_RELAY_DOMAINS "=$mydestination to update "
		     "fast-flush logfile for domain \"%s\"", site);
	status = mail_command_client(MAIL_CLASS_PUBLIC, var_flush_service,
				SEND_ATTR_STR(MAIL_ATTR_REQ, FLUSH_REQ_ADD),
				     SEND_ATTR_STR(MAIL_ATTR_SITE, site),
				 SEND_ATTR_STR(MAIL_ATTR_QUEUEID, queue_id),
				     ATTR_TYPE_END);
    } else if (flush_domains->error == 0)
	status = FLUSH_STAT_DENY;
    else
	status = FLUSH_STAT_FAIL;

    if (msg_verbose)
	msg_info("%s: site %s id %s status %d", myname, site, queue_id,
		 status);

    return (status);
}
