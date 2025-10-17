/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

#ifdef USE_TLS

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <attr.h>
#include <attr_clnt.h>
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>

/* TLS library. */
#include <tls_mgr.h>

/* Application-specific. */

#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)

static ATTR_CLNT *tls_mgr;

/* tls_mgr_open - create client handle */

static void tls_mgr_open(void)
{
    char   *service;

    /*
     * Sanity check.
     */
    if (tls_mgr != 0)
	msg_panic("tls_mgr_open: multiple initialization");

    /*
     * Use whatever IPC is preferred for internal use: UNIX-domain sockets or
     * Solaris streams.
     */
    service = concatenate("local:" TLS_MGR_CLASS "/", var_tls_mgr_service,
			  (char *) 0);
    tls_mgr = attr_clnt_create(service, var_ipc_timeout,
			       var_ipc_idle_limit, var_ipc_ttl_limit);
    myfree(service);

    attr_clnt_control(tls_mgr,
		      ATTR_CLNT_CTL_PROTO, attr_vprint, attr_vscan,
		      ATTR_CLNT_CTL_END);
}

/* tls_mgr_seed - request PRNG seed */

int     tls_mgr_seed(VSTRING *buf, int len)
{
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    /*
     * Request seed.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request attributes */
			  SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_SEED),
			  SEND_ATTR_INT(TLS_MGR_ATTR_SIZE, len),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  RECV_ATTR_DATA(TLS_MGR_ATTR_SEED, buf),
			  ATTR_TYPE_END) != 2)
	status = TLS_MGR_STAT_FAIL;
    return (status);
}

/* tls_mgr_policy - request caching policy */

int     tls_mgr_policy(const char *cache_type, int *cachable, int *timeout)
{
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    /*
     * Request policy.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request attributes */
			SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_POLICY),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_TYPE, cache_type),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply attributes */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  RECV_ATTR_INT(TLS_MGR_ATTR_CACHABLE, cachable),
			  RECV_ATTR_INT(TLS_MGR_ATTR_SESSTOUT, timeout),
			  ATTR_TYPE_END) != 3)
	status = TLS_MGR_STAT_FAIL;
    return (status);
}

/* tls_mgr_lookup - request cached session */

int     tls_mgr_lookup(const char *cache_type, const char *cache_id,
		               VSTRING *buf)
{
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    /*
     * Send the request and receive the reply.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request */
			SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_LOOKUP),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_TYPE, cache_type),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_ID, cache_id),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  RECV_ATTR_DATA(TLS_MGR_ATTR_SESSION, buf),
			  ATTR_TYPE_END) != 2)
	status = TLS_MGR_STAT_FAIL;
    return (status);
}

/* tls_mgr_update - save session to cache */

int     tls_mgr_update(const char *cache_type, const char *cache_id,
		               const char *buf, ssize_t len)
{
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    /*
     * Send the request and receive the reply.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request */
			SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_UPDATE),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_TYPE, cache_type),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_ID, cache_id),
			  SEND_ATTR_DATA(TLS_MGR_ATTR_SESSION, len, buf),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  ATTR_TYPE_END) != 1)
	status = TLS_MGR_STAT_FAIL;
    return (status);
}

/* tls_mgr_delete - remove cached session */

int     tls_mgr_delete(const char *cache_type, const char *cache_id)
{
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    /*
     * Send the request and receive the reply.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request */
			SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_DELETE),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_TYPE, cache_type),
			  SEND_ATTR_STR(TLS_MGR_ATTR_CACHE_ID, cache_id),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  ATTR_TYPE_END) != 1)
	status = TLS_MGR_STAT_FAIL;
    return (status);
}

/* request_scache_key - ask tlsmgr(8) for matching key */

static TLS_TICKET_KEY *request_scache_key(unsigned char *keyname)
{
    TLS_TICKET_KEY tmp;
    static VSTRING *keybuf;
    char   *name;
    size_t  len;
    int     status;

    /*
     * Create the tlsmgr client handle.
     */
    if (tls_mgr == 0)
	tls_mgr_open();

    if (keybuf == 0)
	keybuf = vstring_alloc(sizeof(tmp));

    /* In tlsmgr requests we encode null key names as empty strings. */
    name = keyname ? (char *) keyname : "";
    len = keyname ? TLS_TICKET_NAMELEN : 0;

    /*
     * Send the request and receive the reply.
     */
    if (attr_clnt_request(tls_mgr,
			  ATTR_FLAG_NONE,	/* Request */
			SEND_ATTR_STR(TLS_MGR_ATTR_REQ, TLS_MGR_REQ_TKTKEY),
			  SEND_ATTR_DATA(TLS_MGR_ATTR_KEYNAME, len, name),
			  ATTR_TYPE_END,
			  ATTR_FLAG_MISSING,	/* Reply */
			  RECV_ATTR_INT(TLS_MGR_ATTR_STATUS, &status),
			  RECV_ATTR_DATA(TLS_MGR_ATTR_KEYBUF, keybuf),
			  ATTR_TYPE_END) != 2
	|| status != TLS_MGR_STAT_OK
	|| LEN(keybuf) != sizeof(tmp))
	return (0);

    memcpy((void *) &tmp, STR(keybuf), sizeof(tmp));
    return (tls_scache_key_rotate(&tmp));
}

/* tls_mgr_key - session ticket key lookup, local cache, then tlsmgr(8) */

TLS_TICKET_KEY *tls_mgr_key(unsigned char *keyname, int timeout)
{
    TLS_TICKET_KEY *key = 0;
    time_t  now = time((time_t *) 0);

    /* A zero timeout disables session tickets. */
    if (timeout <= 0)
	return (0);

    if ((key = tls_scache_key(keyname, now, timeout)) == 0)
	key = request_scache_key(keyname);
    return (key);
}

#ifdef TEST

/* System library. */

#include <stdlib.h>

/* Utility library. */

#include <argv.h>
#include <msg_vstream.h>
#include <vstring_vstream.h>
#include <hex_code.h>

/* Global library. */

#include <config.h>

/* Application-specific. */

int     main(int unused_ac, char **av)
{
    VSTRING *inbuf = vstring_alloc(10);
    int     status;
    ARGV   *argv = 0;

    msg_vstream_init(av[0], VSTREAM_ERR);

    msg_verbose = 3;

    mail_conf_read();
    msg_info("using config files in %s", var_config_dir);

    if (chdir(var_queue_dir) < 0)
	msg_fatal("chdir %s: %m", var_queue_dir);

    while (vstring_fgets_nonl(inbuf, VSTREAM_IN)) {
	argv = argv_split(STR(inbuf), CHARS_SPACE);
	if (argv->argc == 0) {
	    argv_free(argv);
	    continue;
	}
#define COMMAND(argv, str, len) \
    (strcasecmp(argv->argv[0], str) == 0 && argv->argc == len)

	if (COMMAND(argv, "policy", 2)) {
	    int     cachable;
	    int     timeout;

	    status = tls_mgr_policy(argv->argv[1], &cachable, &timeout);
	    vstream_printf("status=%d cachable=%d timeout=%d\n",
			   status, cachable, timeout);
	} else if (COMMAND(argv, "seed", 2)) {
	    VSTRING *buf = vstring_alloc(10);
	    VSTRING *hex = vstring_alloc(10);
	    int     len = atoi(argv->argv[1]);

	    status = tls_mgr_seed(buf, len);
	    hex_encode(hex, STR(buf), LEN(buf));
	    vstream_printf("status=%d seed=%s\n", status, STR(hex));
	    vstring_free(hex);
	    vstring_free(buf);
	} else if (COMMAND(argv, "lookup", 3)) {
	    VSTRING *buf = vstring_alloc(10);

	    status = tls_mgr_lookup(argv->argv[1], argv->argv[2], buf);
	    vstream_printf("status=%d session=%.*s\n",
			   status, LEN(buf), STR(buf));
	    vstring_free(buf);
	} else if (COMMAND(argv, "update", 4)) {
	    status = tls_mgr_update(argv->argv[1], argv->argv[2],
				    argv->argv[3], strlen(argv->argv[3]));
	    vstream_printf("status=%d\n", status);
	} else if (COMMAND(argv, "delete", 3)) {
	    status = tls_mgr_delete(argv->argv[1], argv->argv[2]);
	    vstream_printf("status=%d\n", status);
	} else {
	    vstream_printf("usage:\n"
			   "seed byte_count\n"
			   "policy smtpd|smtp|lmtp\n"
			   "lookup smtpd|smtp|lmtp cache_id\n"
			   "update smtpd|smtp|lmtp cache_id session\n"
			   "delete smtpd|smtp|lmtp cache_id\n");
	}
	vstream_fflush(VSTREAM_OUT);
	argv_free(argv);
    }

    vstring_free(inbuf);
    return (0);
}

#endif					/* TEST */

#endif					/* USE_TLS */
