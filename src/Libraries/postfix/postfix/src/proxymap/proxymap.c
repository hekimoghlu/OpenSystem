/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>
#include <htable.h>
#include <stringops.h>
#include <dict.h>

/* Global library. */

#include <mail_conf.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>
#include <dict_proxy.h>

/* Server skeleton. */

#include <mail_server.h>

/* Application-specific. */

 /*
  * XXX All but the last are needed here so that $name expansion dependencies
  * aren't too broken. The fix is to gather all parameter default settings in
  * one place.
  */
char   *var_alias_maps;
char   *var_local_rcpt_maps;
char   *var_virt_alias_maps;
char   *var_virt_alias_doms;
char   *var_virt_mailbox_maps;
char   *var_virt_mailbox_doms;
char   *var_relay_rcpt_maps;
char   *var_relay_domains;
char   *var_canonical_maps;
char   *var_send_canon_maps;
char   *var_rcpt_canon_maps;
char   *var_relocated_maps;
char   *var_transport_maps;
char   *var_verify_map;
char   *var_smtpd_snd_auth_maps;
char   *var_psc_cache_map;
char   *var_proxy_read_maps;
char   *var_proxy_write_maps;

 /*
  * The pre-approved, pre-parsed list of maps.
  */
static HTABLE *proxy_auth_maps;

 /*
  * Shared and static to reduce memory allocation overhead.
  */
static VSTRING *request;
static VSTRING *request_map;
static VSTRING *request_key;
static VSTRING *request_value;
static VSTRING *map_type_name_flags;

 /*
  * Are we a proxy writer or not?
  */
static int proxy_writer;

 /*
  * Silly little macros.
  */
#define STR(x)			vstring_str(x)
#define VSTREQ(x,y)		(strcmp(STR(x),y) == 0)

/* proxy_map_find - look up or open table */

static DICT *proxy_map_find(const char *map_type_name, int request_flags,
			            int *statp)
{
    DICT   *dict;

#define PROXY_COLON	DICT_TYPE_PROXY ":"
#define PROXY_COLON_LEN	(sizeof(PROXY_COLON) - 1)
#define READ_OPEN_FLAGS	O_RDONLY
#define WRITE_OPEN_FLAGS (O_RDWR | O_CREAT)

    /*
     * Canonicalize the map name. If the map is not on the approved list,
     * deny the request.
     */
#define PROXY_MAP_FIND_ERROR_RETURN(x)  { *statp = (x); return (0); }

    while (strncmp(map_type_name, PROXY_COLON, PROXY_COLON_LEN) == 0)
	map_type_name += PROXY_COLON_LEN;
    /* XXX The following breaks with maps that have ':' in their name. */
    if (strchr(map_type_name, ':') == 0)
	PROXY_MAP_FIND_ERROR_RETURN(PROXY_STAT_BAD);
    if (htable_locate(proxy_auth_maps, map_type_name) == 0) {
	msg_warn("request for unapproved table: \"%s\"", map_type_name);
	msg_warn("to approve this table for %s access, list %s:%s in %s:%s",
		 proxy_writer == 0 ? "read-only" : "read-write",
		 DICT_TYPE_PROXY, map_type_name, MAIN_CONF_FILE,
		 proxy_writer == 0 ? VAR_PROXY_READ_MAPS :
		 VAR_PROXY_WRITE_MAPS);
	PROXY_MAP_FIND_ERROR_RETURN(PROXY_STAT_DENY);
    }

    /*
     * Open one instance of a map for each combination of name+flags.
     * 
     * Assume that a map instance can be shared among clients with different
     * paranoia flag settings and with different map lookup flag settings.
     * 
     * XXX The open() flags are passed implicitly, via the selection of the
     * service name. For a more sophisticated interface, appropriate subsets
     * of open() flags should be received directly from the client.
     */
    vstring_sprintf(map_type_name_flags, "%s:%s", map_type_name,
		    dict_flags_str(request_flags & DICT_FLAG_INST_MASK));
    if (msg_verbose)
	msg_info("proxy_map_find: %s", STR(map_type_name_flags));
    if ((dict = dict_handle(STR(map_type_name_flags))) == 0) {
	dict = dict_open(map_type_name, proxy_writer ?
			 WRITE_OPEN_FLAGS : READ_OPEN_FLAGS,
			 request_flags);
	if (dict == 0)
	    msg_panic("proxy_map_find: dict_open null result");
	dict_register(STR(map_type_name_flags), dict);
    }
    dict->error = 0;
    return (dict);
}

/* proxymap_sequence_service - remote sequence service */

static void proxymap_sequence_service(VSTREAM *client_stream)
{
    int     request_flags;
    DICT   *dict;
    int     request_func;
    const char *reply_key;
    const char *reply_value;
    int     dict_status;
    int     reply_status;

    /*
     * Process the request.
     */
    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_TABLE, request_map),
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request_flags),
		  RECV_ATTR_INT(MAIL_ATTR_FUNC, &request_func),
		  ATTR_TYPE_END) != 3
	|| (request_func != DICT_SEQ_FUN_FIRST
	    && request_func != DICT_SEQ_FUN_NEXT)) {
	reply_status = PROXY_STAT_BAD;
	reply_key = reply_value = "";
    } else if ((dict = proxy_map_find(STR(request_map), request_flags,
				      &reply_status)) == 0) {
	reply_key = reply_value = "";
    } else {
	dict->flags = ((dict->flags & ~DICT_FLAG_RQST_MASK)
		       | (request_flags & DICT_FLAG_RQST_MASK));
	dict_status = dict_seq(dict, request_func, &reply_key, &reply_value);
	if (dict_status == 0) {
	    reply_status = PROXY_STAT_OK;
	} else if (dict->error == 0) {
	    reply_status = PROXY_STAT_NOKEY;
	    reply_key = reply_value = "";
	} else {
	    reply_status = (dict->error == DICT_ERR_RETRY ?
			    PROXY_STAT_RETRY : PROXY_STAT_CONFIG);
	    reply_key = reply_value = "";
	}
    }

    /*
     * Respond to the client.
     */
    attr_print(client_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, reply_status),
	       SEND_ATTR_STR(MAIL_ATTR_KEY, reply_key),
	       SEND_ATTR_STR(MAIL_ATTR_VALUE, reply_value),
	       ATTR_TYPE_END);
}

/* proxymap_lookup_service - remote lookup service */

static void proxymap_lookup_service(VSTREAM *client_stream)
{
    int     request_flags;
    DICT   *dict;
    const char *reply_value;
    int     reply_status;

    /*
     * Process the request.
     */
    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_TABLE, request_map),
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request_flags),
		  RECV_ATTR_STR(MAIL_ATTR_KEY, request_key),
		  ATTR_TYPE_END) != 3) {
	reply_status = PROXY_STAT_BAD;
	reply_value = "";
    } else if ((dict = proxy_map_find(STR(request_map), request_flags,
				      &reply_status)) == 0) {
	reply_value = "";
    } else if (dict->flags = ((dict->flags & ~DICT_FLAG_RQST_MASK)
			      | (request_flags & DICT_FLAG_RQST_MASK)),
	       (reply_value = dict_get(dict, STR(request_key))) != 0) {
	reply_status = PROXY_STAT_OK;
    } else if (dict->error == 0) {
	reply_status = PROXY_STAT_NOKEY;
	reply_value = "";
    } else {
	reply_status = (dict->error == DICT_ERR_RETRY ?
			PROXY_STAT_RETRY : PROXY_STAT_CONFIG);
	reply_value = "";
    }

    /*
     * Respond to the client.
     */
    attr_print(client_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, reply_status),
	       SEND_ATTR_STR(MAIL_ATTR_VALUE, reply_value),
	       ATTR_TYPE_END);
}

/* proxymap_update_service - remote update service */

static void proxymap_update_service(VSTREAM *client_stream)
{
    int     request_flags;
    DICT   *dict;
    int     dict_status;
    int     reply_status;

    /*
     * Process the request.
     * 
     * XXX We don't close maps, so we must turn on synchronous update to ensure
     * that the on-disk data is in a consistent state between updates.
     * 
     * XXX We ignore duplicates, because the proxymap server would abort
     * otherwise.
     */
    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_TABLE, request_map),
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request_flags),
		  RECV_ATTR_STR(MAIL_ATTR_KEY, request_key),
		  RECV_ATTR_STR(MAIL_ATTR_VALUE, request_value),
		  ATTR_TYPE_END) != 4) {
	reply_status = PROXY_STAT_BAD;
    } else if (proxy_writer == 0) {
	msg_warn("refusing %s update request on non-%s service",
		 STR(request_map), MAIL_SERVICE_PROXYWRITE);
	reply_status = PROXY_STAT_DENY;
    } else if ((dict = proxy_map_find(STR(request_map), request_flags,
				      &reply_status)) == 0) {
	 /* void */ ;
    } else {
	dict->flags = ((dict->flags & ~DICT_FLAG_RQST_MASK)
		       | (request_flags & DICT_FLAG_RQST_MASK)
		       | DICT_FLAG_SYNC_UPDATE | DICT_FLAG_DUP_REPLACE);
	dict_status = dict_put(dict, STR(request_key), STR(request_value));
	if (dict_status == 0) {
	    reply_status = PROXY_STAT_OK;
	} else if (dict->error == 0) {
	    reply_status = PROXY_STAT_NOKEY;
	} else {
	    reply_status = (dict->error == DICT_ERR_RETRY ?
			    PROXY_STAT_RETRY : PROXY_STAT_CONFIG);
	}
    }

    /*
     * Respond to the client.
     */
    attr_print(client_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, reply_status),
	       ATTR_TYPE_END);
}

/* proxymap_delete_service - remote delete service */

static void proxymap_delete_service(VSTREAM *client_stream)
{
    int     request_flags;
    DICT   *dict;
    int     dict_status;
    int     reply_status;

    /*
     * Process the request.
     * 
     * XXX We don't close maps, so we must turn on synchronous update to ensure
     * that the on-disk data is in a consistent state between updates.
     */
    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_TABLE, request_map),
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request_flags),
		  RECV_ATTR_STR(MAIL_ATTR_KEY, request_key),
		  ATTR_TYPE_END) != 3) {
	reply_status = PROXY_STAT_BAD;
    } else if (proxy_writer == 0) {
	msg_warn("refusing %s delete request on non-%s service",
		 STR(request_map), MAIL_SERVICE_PROXYWRITE);
	reply_status = PROXY_STAT_DENY;
    } else if ((dict = proxy_map_find(STR(request_map), request_flags,
				      &reply_status)) == 0) {
	 /* void */ ;
    } else {
	dict->flags = ((dict->flags & ~DICT_FLAG_RQST_MASK)
		       | (request_flags & DICT_FLAG_RQST_MASK)
		       | DICT_FLAG_SYNC_UPDATE);
	dict_status = dict_del(dict, STR(request_key));
	if (dict_status == 0) {
	    reply_status = PROXY_STAT_OK;
	} else if (dict->error == 0) {
	    reply_status = PROXY_STAT_NOKEY;
	} else {
	    reply_status = (dict->error == DICT_ERR_RETRY ?
			    PROXY_STAT_RETRY : PROXY_STAT_CONFIG);
	}
    }

    /*
     * Respond to the client.
     */
    attr_print(client_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, reply_status),
	       ATTR_TYPE_END);
}

/* proxymap_open_service - open remote lookup table */

static void proxymap_open_service(VSTREAM *client_stream)
{
    int     request_flags;
    DICT   *dict;
    int     reply_status;
    int     reply_flags;

    /*
     * Process the request.
     */
    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_TABLE, request_map),
		  RECV_ATTR_INT(MAIL_ATTR_FLAGS, &request_flags),
		  ATTR_TYPE_END) != 2) {
	reply_status = PROXY_STAT_BAD;
	reply_flags = 0;
    } else if ((dict = proxy_map_find(STR(request_map), request_flags,
				      &reply_status)) == 0) {
	reply_flags = 0;
    } else {
	reply_status = PROXY_STAT_OK;
	reply_flags = dict->flags;
    }

    /*
     * Respond to the client.
     */
    attr_print(client_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, reply_status),
	       SEND_ATTR_INT(MAIL_ATTR_FLAGS, reply_flags),
	       ATTR_TYPE_END);
}

/* proxymap_service - perform service for client */

static void proxymap_service(VSTREAM *client_stream, char *unused_service,
			             char **argv)
{

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Deadline enforcement.
     */
    if (vstream_fstat(client_stream, VSTREAM_FLAG_DEADLINE) == 0)
	vstream_control(client_stream,
			CA_VSTREAM_CTL_TIMEOUT(1),
			CA_VSTREAM_CTL_END);

    /*
     * This routine runs whenever a client connects to the socket dedicated
     * to the proxymap service. All connection-management stuff is handled by
     * the common code in multi_server.c.
     */
    vstream_control(client_stream,
		    CA_VSTREAM_CTL_START_DEADLINE,
		    CA_VSTREAM_CTL_END);
    if (attr_scan(client_stream,
		  ATTR_FLAG_MORE | ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_REQ, request),
		  ATTR_TYPE_END) == 1) {
	if (VSTREQ(request, PROXY_REQ_LOOKUP)) {
	    proxymap_lookup_service(client_stream);
	} else if (VSTREQ(request, PROXY_REQ_UPDATE)) {
	    proxymap_update_service(client_stream);
	} else if (VSTREQ(request, PROXY_REQ_DELETE)) {
	    proxymap_delete_service(client_stream);
	} else if (VSTREQ(request, PROXY_REQ_SEQUENCE)) {
	    proxymap_sequence_service(client_stream);
	} else if (VSTREQ(request, PROXY_REQ_OPEN)) {
	    proxymap_open_service(client_stream);
	} else {
	    msg_warn("unrecognized request: \"%s\", ignored", STR(request));
	    attr_print(client_stream, ATTR_FLAG_NONE,
		       SEND_ATTR_INT(MAIL_ATTR_STATUS, PROXY_STAT_BAD),
		       ATTR_TYPE_END);
	}
    }
    vstream_control(client_stream,
		    CA_VSTREAM_CTL_START_DEADLINE,
		    CA_VSTREAM_CTL_END);
    vstream_fflush(client_stream);
}

/* dict_proxy_open - intercept remote map request from inside library */

DICT   *dict_proxy_open(const char *map, int open_flags, int dict_flags)
{
    if (msg_verbose)
	msg_info("dict_proxy_open(%s, 0%o, 0%o) called from internal routine",
		 map, open_flags, dict_flags);
    while (strncmp(map, PROXY_COLON, PROXY_COLON_LEN) == 0)
	map += PROXY_COLON_LEN;
    return (dict_open(map, open_flags, dict_flags));
}

/* post_jail_init - initialization after privilege drop */

static void post_jail_init(char *service_name, char **unused_argv)
{
    const char *sep = CHARS_COMMA_SP;
    const char *parens = CHARS_BRACE;
    char   *saved_filter;
    char   *bp;
    char   *type_name;

    /*
     * Are we proxy writer?
     */
    if (strcmp(service_name, MAIL_SERVICE_PROXYWRITE) == 0)
	proxy_writer = 1;
    else if (strcmp(service_name, MAIL_SERVICE_PROXYMAP) != 0)
	msg_fatal("service name must be one of %s or %s",
		  MAIL_SERVICE_PROXYMAP, MAIL_SERVICE_PROXYMAP);

    /*
     * Pre-allocate buffers.
     */
    request = vstring_alloc(10);
    request_map = vstring_alloc(10);
    request_key = vstring_alloc(10);
    request_value = vstring_alloc(10);
    map_type_name_flags = vstring_alloc(10);

    /*
     * Prepare the pre-approved list of proxied tables.
     */
    saved_filter = bp = mystrdup(proxy_writer ? var_proxy_write_maps :
				 var_proxy_read_maps);
    proxy_auth_maps = htable_create(13);
    while ((type_name = mystrtokq(&bp, sep, parens)) != 0) {
	if (strncmp(type_name, PROXY_COLON, PROXY_COLON_LEN))
	    continue;
	do {
	    type_name += PROXY_COLON_LEN;
	} while (!strncmp(type_name, PROXY_COLON, PROXY_COLON_LEN));
	if (strchr(type_name, ':') != 0
	    && htable_locate(proxy_auth_maps, type_name) == 0)
	    (void) htable_enter(proxy_auth_maps, type_name, (void *) 0);
    }
    myfree(saved_filter);

    /*
     * Never, ever, get killed by a master signal, as that could corrupt a
     * persistent database when we're in the middle of an update.
     */
    if (proxy_writer != 0)
	setsid();
}

/* pre_accept - see if tables have changed */

static void pre_accept(char *unused_name, char **unused_argv)
{
    const char *table;

    if (proxy_writer == 0 && (table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_ALIAS_MAPS, DEF_ALIAS_MAPS, &var_alias_maps, 0, 0,
	VAR_LOCAL_RCPT_MAPS, DEF_LOCAL_RCPT_MAPS, &var_local_rcpt_maps, 0, 0,
	VAR_VIRT_ALIAS_MAPS, DEF_VIRT_ALIAS_MAPS, &var_virt_alias_maps, 0, 0,
	VAR_VIRT_ALIAS_DOMS, DEF_VIRT_ALIAS_DOMS, &var_virt_alias_doms, 0, 0,
	VAR_VIRT_MAILBOX_MAPS, DEF_VIRT_MAILBOX_MAPS, &var_virt_mailbox_maps, 0, 0,
	VAR_VIRT_MAILBOX_DOMS, DEF_VIRT_MAILBOX_DOMS, &var_virt_mailbox_doms, 0, 0,
	VAR_RELAY_RCPT_MAPS, DEF_RELAY_RCPT_MAPS, &var_relay_rcpt_maps, 0, 0,
	VAR_RELAY_DOMAINS, DEF_RELAY_DOMAINS, &var_relay_domains, 0, 0,
	VAR_CANONICAL_MAPS, DEF_CANONICAL_MAPS, &var_canonical_maps, 0, 0,
	VAR_SEND_CANON_MAPS, DEF_SEND_CANON_MAPS, &var_send_canon_maps, 0, 0,
	VAR_RCPT_CANON_MAPS, DEF_RCPT_CANON_MAPS, &var_rcpt_canon_maps, 0, 0,
	VAR_RELOCATED_MAPS, DEF_RELOCATED_MAPS, &var_relocated_maps, 0, 0,
	VAR_TRANSPORT_MAPS, DEF_TRANSPORT_MAPS, &var_transport_maps, 0, 0,
	VAR_VERIFY_MAP, DEF_VERIFY_MAP, &var_verify_map, 0, 0,
	VAR_SMTPD_SND_AUTH_MAPS, DEF_SMTPD_SND_AUTH_MAPS, &var_smtpd_snd_auth_maps, 0, 0,
	VAR_PSC_CACHE_MAP, DEF_PSC_CACHE_MAP, &var_psc_cache_map, 0, 0,
	/* The following two must be last for $mapname to work as expected. */
	VAR_PROXY_READ_MAPS, DEF_PROXY_READ_MAPS, &var_proxy_read_maps, 0, 0,
	VAR_PROXY_WRITE_MAPS, DEF_PROXY_WRITE_MAPS, &var_proxy_write_maps, 0, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    multi_server_main(argc, argv, proxymap_service,
		      CA_MAIL_SERVER_STR_TABLE(str_table),
		      CA_MAIL_SERVER_POST_INIT(post_jail_init),
		      CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
    /* XXX CA_MAIL_SERVER_SOLITARY if proxywrite */
		      0);
}
