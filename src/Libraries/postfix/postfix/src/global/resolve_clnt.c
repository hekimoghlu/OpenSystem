/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include <sys_defs.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>
#include <events.h>
#include <iostuff.h>

/* Global library. */

#include "mail_proto.h"
#include "mail_params.h"
#include "clnt_stream.h"
#include "resolve_clnt.h"

/* Application-specific. */

 /*
  * XXX this is shared with the rewrite client to save a file descriptor.
  */
extern CLNT_STREAM *rewrite_clnt_stream;

static time_t last_expire;
static VSTRING *last_class;
static VSTRING *last_sender;
static VSTRING *last_addr;
static RESOLVE_REPLY last_reply;

/* resolve_clnt_init - initialize reply */

void    resolve_clnt_init(RESOLVE_REPLY *reply)
{
    reply->transport = vstring_alloc(100);
    reply->nexthop = vstring_alloc(100);
    reply->recipient = vstring_alloc(100);
    reply->flags = 0;
}

/* resolve_clnt - resolve address to (transport, next hop, recipient) */

void    resolve_clnt(const char *class, const char *sender,
		             const char *addr, RESOLVE_REPLY *reply)
{
    const char *myname = "resolve_clnt";
    VSTREAM *stream;
    int     server_flags;
    int     count = 0;

    /*
     * One-entry cache.
     */
    if (last_addr == 0) {
	last_class = vstring_alloc(10);
	last_sender = vstring_alloc(10);
	last_addr = vstring_alloc(100);
	resolve_clnt_init(&last_reply);
    }

    /*
     * Sanity check. The result must not clobber the input because we may
     * have to retransmit the request.
     */
#define STR vstring_str

    if (addr == STR(reply->recipient))
	msg_panic("%s: result clobbers input", myname);

    /*
     * Peek at the cache.
     */
#define IFSET(flag, text) ((reply->flags & (flag)) ? (text) : "")

    if (time((time_t *) 0) < last_expire
	&& *addr && strcmp(addr, STR(last_addr)) == 0
	&& strcmp(class, STR(last_class)) == 0
	&& strcmp(sender, STR(last_sender)) == 0) {
	vstring_strcpy(reply->transport, STR(last_reply.transport));
	vstring_strcpy(reply->nexthop, STR(last_reply.nexthop));
	vstring_strcpy(reply->recipient, STR(last_reply.recipient));
	reply->flags = last_reply.flags;
	if (msg_verbose)
	    msg_info("%s: cached: `%s' -> `%s' -> transp=`%s' host=`%s' rcpt=`%s' flags=%s%s%s%s class=%s%s%s%s%s",
		     myname, sender, addr, STR(reply->transport),
		     STR(reply->nexthop), STR(reply->recipient),
		     IFSET(RESOLVE_FLAG_FINAL, "final"),
		     IFSET(RESOLVE_FLAG_ROUTED, "routed"),
		     IFSET(RESOLVE_FLAG_ERROR, "error"),
		     IFSET(RESOLVE_FLAG_FAIL, "fail"),
		     IFSET(RESOLVE_CLASS_LOCAL, "local"),
		     IFSET(RESOLVE_CLASS_ALIAS, "alias"),
		     IFSET(RESOLVE_CLASS_VIRTUAL, "virtual"),
		     IFSET(RESOLVE_CLASS_RELAY, "relay"),
		     IFSET(RESOLVE_CLASS_DEFAULT, "default"));
	return;
    }

    /*
     * Keep trying until we get a complete response. The resolve service is
     * CPU bound; making the client asynchronous would just complicate the
     * code.
     */
    if (rewrite_clnt_stream == 0)
	rewrite_clnt_stream = clnt_stream_create(MAIL_CLASS_PRIVATE,
						 var_rewrite_service,
						 var_ipc_idle_limit,
						 var_ipc_ttl_limit);

    for (;;) {
	stream = clnt_stream_access(rewrite_clnt_stream);
	errno = 0;
	count += 1;
	if (attr_print(stream, ATTR_FLAG_NONE,
		       SEND_ATTR_STR(MAIL_ATTR_REQ, class),
		       SEND_ATTR_STR(MAIL_ATTR_SENDER, sender),
		       SEND_ATTR_STR(MAIL_ATTR_ADDR, addr),
		       ATTR_TYPE_END) != 0
	    || vstream_fflush(stream)
	    || attr_scan(stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_INT(MAIL_ATTR_FLAGS, &server_flags),
		       RECV_ATTR_STR(MAIL_ATTR_TRANSPORT, reply->transport),
			 RECV_ATTR_STR(MAIL_ATTR_NEXTHOP, reply->nexthop),
			 RECV_ATTR_STR(MAIL_ATTR_RECIP, reply->recipient),
			 RECV_ATTR_INT(MAIL_ATTR_FLAGS, &reply->flags),
			 ATTR_TYPE_END) != 5) {
	    if (msg_verbose || count > 1 || (errno && errno != EPIPE && errno != ENOENT))
		msg_warn("problem talking to service %s: %m",
			 var_rewrite_service);
	} else {
	    if (msg_verbose)
		msg_info("%s: `%s' -> `%s' -> transp=`%s' host=`%s' rcpt=`%s' flags=%s%s%s%s class=%s%s%s%s%s",
			 myname, sender, addr, STR(reply->transport),
			 STR(reply->nexthop), STR(reply->recipient),
			 IFSET(RESOLVE_FLAG_FINAL, "final"),
			 IFSET(RESOLVE_FLAG_ROUTED, "routed"),
			 IFSET(RESOLVE_FLAG_ERROR, "error"),
			 IFSET(RESOLVE_FLAG_FAIL, "fail"),
			 IFSET(RESOLVE_CLASS_LOCAL, "local"),
			 IFSET(RESOLVE_CLASS_ALIAS, "alias"),
			 IFSET(RESOLVE_CLASS_VIRTUAL, "virtual"),
			 IFSET(RESOLVE_CLASS_RELAY, "relay"),
			 IFSET(RESOLVE_CLASS_DEFAULT, "default"));
	    /* Server-requested disconnect. */
	    if (server_flags != 0)
		clnt_stream_recover(rewrite_clnt_stream);
	    if (STR(reply->transport)[0] == 0)
		msg_warn("%s: null transport result for: <%s>", myname, addr);
	    else if (STR(reply->recipient)[0] == 0 && *addr != 0)
		msg_warn("%s: null recipient result for: <%s>", myname, addr);
	    else
		break;
	}
	sleep(1);				/* XXX make configurable */
	clnt_stream_recover(rewrite_clnt_stream);
    }

    /*
     * Update the cache.
     */
    vstring_strcpy(last_class, class);
    vstring_strcpy(last_sender, sender);
    vstring_strcpy(last_addr, addr);
    vstring_strcpy(last_reply.transport, STR(reply->transport));
    vstring_strcpy(last_reply.nexthop, STR(reply->nexthop));
    vstring_strcpy(last_reply.recipient, STR(reply->recipient));
    last_reply.flags = reply->flags;
    last_expire = time((time_t *) 0) + 30;	/* XXX make configurable */
}

/* resolve_clnt_free - destroy reply */

void    resolve_clnt_free(RESOLVE_REPLY *reply)
{
    reply->transport = vstring_free(reply->transport);
    reply->nexthop = vstring_free(reply->nexthop);
    reply->recipient = vstring_free(reply->recipient);
}

#ifdef TEST

#include <stdlib.h>
#include <msg_vstream.h>
#include <vstring_vstream.h>
#include <split_at.h>
#include <mail_conf.h>

static NORETURN usage(char *myname)
{
    msg_fatal("usage: %s [-v] [address...]", myname);
}

static void resolve(char *class, char *addr, RESOLVE_REPLY *reply)
{
    struct RESOLVE_FLAG_TABLE {
	int     flag;
	const char *name;
    };
    struct RESOLVE_FLAG_TABLE resolve_flag_table[] = {
	RESOLVE_FLAG_FINAL, "FLAG_FINAL",
	RESOLVE_FLAG_ROUTED, "FLAG_ROUTED",
	RESOLVE_FLAG_ERROR, "FLAG_ERROR",
	RESOLVE_FLAG_FAIL, "FLAG_FAIL",
	RESOLVE_CLASS_LOCAL, "CLASS_LOCAL",
	RESOLVE_CLASS_ALIAS, "CLASS_ALIAS",
	RESOLVE_CLASS_VIRTUAL, "CLASS_VIRTUAL",
	RESOLVE_CLASS_RELAY, "CLASS_RELAY",
	RESOLVE_CLASS_DEFAULT, "CLASS_DEFAULT",
	0,
    };
    struct RESOLVE_FLAG_TABLE *fp;

    resolve_clnt(class, RESOLVE_NULL_FROM, addr, reply);
    if (reply->flags & RESOLVE_FLAG_FAIL) {
	vstream_printf("request failed\n");
    } else {
	vstream_printf("%-10s %s\n", "class", class);
	vstream_printf("%-10s %s\n", "address", addr);
	vstream_printf("%-10s %s\n", "transport", STR(reply->transport));
	vstream_printf("%-10s %s\n", "nexthop", *STR(reply->nexthop) ?
		       STR(reply->nexthop) : "[none]");
	vstream_printf("%-10s %s\n", "recipient", STR(reply->recipient));
	vstream_printf("%-10s ", "flags");
	for (fp = resolve_flag_table; fp->name; fp++) {
	    if (reply->flags & fp->flag) {
		vstream_printf("%s ", fp->name);
		reply->flags &= ~fp->flag;
	    }
	}
	if (reply->flags != 0)
	    vstream_printf("Unknown flag 0x%x", reply->flags);
	vstream_printf("\n\n");
	vstream_fflush(VSTREAM_OUT);
    }
}

int     main(int argc, char **argv)
{
    RESOLVE_REPLY reply;
    char   *addr;
    int     ch;

    msg_vstream_init(argv[0], VSTREAM_ERR);

    mail_conf_read();
    msg_info("using config files in %s", var_config_dir);
    if (chdir(var_queue_dir) < 0)
	msg_fatal("chdir %s: %m", var_queue_dir);

    while ((ch = GETOPT(argc, argv, "v")) > 0) {
	switch (ch) {
	case 'v':
	    msg_verbose++;
	    break;
	default:
	    usage(argv[0]);
	}
    }
    resolve_clnt_init(&reply);

    if (argc > optind) {
	while (argv[optind] && argv[optind + 1]) {
	    resolve(argv[optind], argv[optind + 1], &reply);
	    optind += 2;
	}
    } else {
	VSTRING *buffer = vstring_alloc(1);

	while (vstring_fgets_nonl(buffer, VSTREAM_IN)) {
	    addr = split_at(STR(buffer), ' ');
	    if (*STR(buffer) == 0)
		msg_fatal("need as input: class [address]");
	    if (addr == 0)
		addr = "";
	    resolve(STR(buffer), addr, &reply);
	}
	vstring_free(buffer);
    }
    resolve_clnt_free(&reply);
    exit(0);
}

#endif
