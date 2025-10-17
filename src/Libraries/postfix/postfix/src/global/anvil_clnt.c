/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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

/* Utility library. */

#include <mymalloc.h>
#include <msg.h>
#include <attr_clnt.h>
#include <stringops.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_params.h>
#include <anvil_clnt.h>

/* Application specific. */

#define ANVIL_IDENT(service, addr) \
    printable(concatenate(service, ":", addr, (char *) 0), '?')

/* anvil_clnt_create - instantiate connection rate service client */

ANVIL_CLNT *anvil_clnt_create(void)
{
    ATTR_CLNT *anvil_clnt;

    /*
     * Use whatever IPC is preferred for internal use: UNIX-domain sockets or
     * Solaris streams.
     */
#ifndef VAR_ANVIL_SERVICE
    anvil_clnt = attr_clnt_create("local:" ANVIL_CLASS "/" ANVIL_SERVICE,
				  var_ipc_timeout, 0, 0);
#else
    anvil_clnt = attr_clnt_create(var_anvil_service, var_ipc_timeout, 0, 0);
#endif
    return ((ANVIL_CLNT *) anvil_clnt);
}

/* anvil_clnt_free - destroy connection rate service client */

void    anvil_clnt_free(ANVIL_CLNT *anvil_clnt)
{
    attr_clnt_free((ATTR_CLNT *) anvil_clnt);
}

/* anvil_clnt_lookup - status query */

int     anvil_clnt_lookup(ANVIL_CLNT *anvil_clnt, const char *service,
			          const char *addr, int *count, int *rate,
		             int *msgs, int *rcpts, int *newtls, int *auths)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_LOOKUP),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_COUNT, count),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, rate),
			  RECV_ATTR_INT(ANVIL_ATTR_MAIL, msgs),
			  RECV_ATTR_INT(ANVIL_ATTR_RCPT, rcpts),
			  RECV_ATTR_INT(ANVIL_ATTR_NTLS, newtls),
			  RECV_ATTR_INT(ANVIL_ATTR_AUTH, auths),
			  ATTR_TYPE_END) != 7)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_connect - heads-up and status query */

int     anvil_clnt_connect(ANVIL_CLNT *anvil_clnt, const char *service,
			           const char *addr, int *count, int *rate)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_CONN),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_COUNT, count),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, rate),
			  ATTR_TYPE_END) != 3)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_mail - heads-up and status query */

int     anvil_clnt_mail(ANVIL_CLNT *anvil_clnt, const char *service,
			        const char *addr, int *msgs)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_MAIL),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, msgs),
			  ATTR_TYPE_END) != 2)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_rcpt - heads-up and status query */

int     anvil_clnt_rcpt(ANVIL_CLNT *anvil_clnt, const char *service,
			        const char *addr, int *rcpts)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_RCPT),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, rcpts),
			  ATTR_TYPE_END) != 2)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_newtls - heads-up and status query */

int     anvil_clnt_newtls(ANVIL_CLNT *anvil_clnt, const char *service,
			          const char *addr, int *newtls)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_NTLS),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, newtls),
			  ATTR_TYPE_END) != 2)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_newtls_stat - status query */

int     anvil_clnt_newtls_stat(ANVIL_CLNT *anvil_clnt, const char *service,
			               const char *addr, int *newtls)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_NTLS_STAT),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, newtls),
			  ATTR_TYPE_END) != 2)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_auth - heads-up and status query */

int     anvil_clnt_auth(ANVIL_CLNT *anvil_clnt, const char *service,
			        const char *addr, int *auths)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_AUTH),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  RECV_ATTR_INT(ANVIL_ATTR_RATE, auths),
			  ATTR_TYPE_END) != 2)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

/* anvil_clnt_disconnect - heads-up only */

int     anvil_clnt_disconnect(ANVIL_CLNT *anvil_clnt, const char *service,
			              const char *addr)
{
    char   *ident = ANVIL_IDENT(service, addr);
    int     status;

    if (attr_clnt_request((ATTR_CLNT *) anvil_clnt,
			  ATTR_FLAG_NONE,	/* Query attributes. */
			  SEND_ATTR_STR(ANVIL_ATTR_REQ, ANVIL_REQ_DISC),
			  SEND_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes. */
			  RECV_ATTR_INT(ANVIL_ATTR_STATUS, &status),
			  ATTR_TYPE_END) != 1)
	status = ANVIL_STAT_FAIL;
    else if (status != ANVIL_STAT_OK)
	status = ANVIL_STAT_FAIL;
    myfree(ident);
    return (status);
}

#ifdef TEST

 /*
  * Stand-alone client for testing.
  */
#include <unistd.h>
#include <string.h>
#include <msg_vstream.h>
#include <mail_conf.h>
#include <mail_params.h>
#include <vstring_vstream.h>

static void usage(void)
{
    vstream_printf("usage: "
		   ANVIL_REQ_CONN " service addr | "
		   ANVIL_REQ_DISC " service addr | "
		   ANVIL_REQ_MAIL " service addr | "
		   ANVIL_REQ_RCPT " service addr | "
		   ANVIL_REQ_NTLS " service addr | "
		   ANVIL_REQ_NTLS_STAT " service addr | "
		   ANVIL_REQ_AUTH " service addr | "
		   ANVIL_REQ_LOOKUP " service addr\n");
}

int     main(int unused_argc, char **argv)
{
    VSTRING *inbuf = vstring_alloc(1);
    char   *bufp;
    char   *cmd;
    ssize_t cmd_len;
    char   *service;
    char   *addr;
    int     count;
    int     rate;
    int     msgs;
    int     rcpts;
    int     newtls;
    int     auths;
    ANVIL_CLNT *anvil;

    msg_vstream_init(argv[0], VSTREAM_ERR);

    mail_conf_read();
    msg_info("using config files in %s", var_config_dir);
    if (chdir(var_queue_dir) < 0)
	msg_fatal("chdir %s: %m", var_queue_dir);

    msg_verbose++;

    anvil = anvil_clnt_create();

    while (vstring_fgets_nonl(inbuf, VSTREAM_IN)) {
	bufp = vstring_str(inbuf);
	if ((cmd = mystrtok(&bufp, " ")) == 0 || *bufp == 0
	    || (service = mystrtok(&bufp, " ")) == 0 || *service == 0
	    || (addr = mystrtok(&bufp, " ")) == 0 || *addr == 0
	    || mystrtok(&bufp, " ") != 0) {
	    vstream_printf("bad command syntax\n");
	    usage();
	    vstream_fflush(VSTREAM_OUT);
	    continue;
	}
	cmd_len = strlen(cmd);
	if (strncmp(cmd, ANVIL_REQ_CONN, cmd_len) == 0) {
	    if (anvil_clnt_connect(anvil, service, addr, &count, &rate) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("count=%d, rate=%d\n", count, rate);
	} else if (strncmp(cmd, ANVIL_REQ_MAIL, cmd_len) == 0) {
	    if (anvil_clnt_mail(anvil, service, addr, &msgs) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("rate=%d\n", msgs);
	} else if (strncmp(cmd, ANVIL_REQ_RCPT, cmd_len) == 0) {
	    if (anvil_clnt_rcpt(anvil, service, addr, &rcpts) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("rate=%d\n", rcpts);
	} else if (strncmp(cmd, ANVIL_REQ_NTLS, cmd_len) == 0) {
	    if (anvil_clnt_newtls(anvil, service, addr, &newtls) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("rate=%d\n", newtls);
	} else if (strncmp(cmd, ANVIL_REQ_AUTH, cmd_len) == 0) {
	    if (anvil_clnt_auth(anvil, service, addr, &auths) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("rate=%d\n", auths);
	} else if (strncmp(cmd, ANVIL_REQ_NTLS_STAT, cmd_len) == 0) {
	    if (anvil_clnt_newtls_stat(anvil, service, addr, &newtls) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("rate=%d\n", newtls);
	} else if (strncmp(cmd, ANVIL_REQ_DISC, cmd_len) == 0) {
	    if (anvil_clnt_disconnect(anvil, service, addr) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("OK\n");
	} else if (strncmp(cmd, ANVIL_REQ_LOOKUP, cmd_len) == 0) {
	    if (anvil_clnt_lookup(anvil, service, addr, &count, &rate, &msgs,
				  &rcpts, &newtls, &auths) != ANVIL_STAT_OK)
		msg_warn("error!");
	    else
		vstream_printf("count=%d, rate=%d msgs=%d rcpts=%d newtls=%d "
			     "auths=%d\n", count, rate, msgs, rcpts, newtls,
			       auths);
	} else {
	    vstream_printf("bad command: \"%s\"\n", cmd);
	    usage();
	}
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(inbuf);
    anvil_clnt_free(anvil);
    return (0);
}

#endif
