/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <dict.h>
#include <stringops.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <name_mask.h>
#include <name_code.h>

/* Global library. */

#include <deliver_request.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <debug_peer.h>
#include <flush_clnt.h>
#include <scache.h>
#include <string_list.h>
#include <maps.h>
#include <ext_prop.h>

/* DNS library. */

#include <dns.h>

/* Single server skeleton. */

#include <mail_server.h>

/* Application-specific. */

#include "smtp.h"
#include "smtp_sasl.h"

 /*
  * Tunable parameters. These have compiled-in defaults that can be overruled
  * by settings in the global Postfix configuration file.
  */
int     var_smtp_conn_tmout;
int     var_smtp_helo_tmout;
int     var_smtp_xfwd_tmout;
int     var_smtp_mail_tmout;
int     var_smtp_rcpt_tmout;
int     var_smtp_data0_tmout;
int     var_smtp_data1_tmout;
int     var_smtp_data2_tmout;
int     var_smtp_rset_tmout;
int     var_smtp_quit_tmout;
char   *var_inet_interfaces;
char   *var_notify_classes;
int     var_smtp_skip_5xx_greeting;
int     var_ign_mx_lookup_err;
int     var_skip_quit_resp;
char   *var_fallback_relay;
char   *var_bestmx_transp;
char   *var_error_rcpt;
int     var_smtp_always_ehlo;
int     var_smtp_never_ehlo;
char   *var_smtp_sasl_opts;
char   *var_smtp_sasl_path;
char   *var_smtp_sasl_passwd;
bool    var_smtp_sasl_enable;
char   *var_smtp_sasl_mechs;
char   *var_smtp_sasl_type;
char   *var_smtp_bind_addr;
char   *var_smtp_bind_addr6;
char   *var_smtp_vrfy_tgt;
bool    var_smtp_rand_addr;
int     var_smtp_pix_thresh;
int     var_queue_run_delay;
int     var_min_backoff_time;
int     var_smtp_pix_delay;
int     var_smtp_line_limit;
char   *var_smtp_helo_name;
char   *var_smtp_host_lookup;
bool    var_smtp_quote_821_env;
bool    var_smtp_defer_mxaddr;
bool    var_smtp_send_xforward;
int     var_smtp_mxaddr_limit;
int     var_smtp_mxsess_limit;
int     var_smtp_cache_conn;
int     var_smtp_reuse_time;
int     var_smtp_reuse_count;
char   *var_smtp_cache_dest;
char   *var_scache_service;		/* You can now leave this here. */
bool    var_smtp_cache_demand;
char   *var_smtp_ehlo_dis_words;
char   *var_smtp_ehlo_dis_maps;
char   *var_smtp_addr_pref;

char   *var_smtp_tls_level;
bool    var_smtp_use_tls;
bool    var_smtp_enforce_tls;
char   *var_smtp_tls_per_site;
char   *var_smtp_tls_policy;
bool    var_smtp_tls_wrappermode;

#ifdef USE_TLS
char   *var_smtp_sasl_tls_opts;
char   *var_smtp_sasl_tlsv_opts;
int     var_smtp_starttls_tmout;
char   *var_smtp_tls_CAfile;
char   *var_smtp_tls_CApath;
char   *var_smtp_tls_cert_file;
char   *var_smtp_tls_mand_ciph;
char   *var_smtp_tls_excl_ciph;
char   *var_smtp_tls_mand_excl;
char   *var_smtp_tls_dcert_file;
char   *var_smtp_tls_dkey_file;
bool    var_smtp_tls_enforce_peername;
char   *var_smtp_tls_key_file;
char   *var_smtp_tls_loglevel;
bool    var_smtp_tls_note_starttls_offer;
char   *var_smtp_tls_mand_proto;
char   *var_smtp_tls_sec_cmatch;
int     var_smtp_tls_scert_vd;
char   *var_smtp_tls_vfy_cmatch;
char   *var_smtp_tls_fpt_cmatch;
char   *var_smtp_tls_fpt_dgst;
char   *var_smtp_tls_tafile;
char   *var_smtp_tls_proto;
char   *var_smtp_tls_ciph;
char   *var_smtp_tls_eccert_file;
char   *var_smtp_tls_eckey_file;
bool    var_smtp_tls_blk_early_mail_reply;
bool    var_smtp_tls_force_tlsa;
char   *var_smtp_tls_insecure_mx_policy;

#endif

char   *var_smtp_generic_maps;
char   *var_prop_extension;
bool    var_smtp_sender_auth;
char   *var_smtp_tcp_port;
int     var_scache_proto_tmout;
bool    var_smtp_cname_overr;
char   *var_smtp_pix_bug_words;
char   *var_smtp_pix_bug_maps;
char   *var_cyrus_conf_path;
char   *var_smtp_head_chks;
char   *var_smtp_mime_chks;
char   *var_smtp_nest_chks;
char   *var_smtp_body_chks;
char   *var_smtp_resp_filter;
bool    var_lmtp_assume_final;
char   *var_smtp_dns_res_opt;
char   *var_smtp_dns_support;
bool    var_smtp_rec_deadline;
bool    var_smtp_dummy_mail_auth;
char   *var_smtp_dsn_filter;
char   *var_smtp_dns_re_filter;

 /* Special handling of 535 AUTH errors. */
char   *var_smtp_sasl_auth_cache_name;
int     var_smtp_sasl_auth_cache_time;
bool    var_smtp_sasl_auth_soft_bounce;

 /*
  * Global variables.
  */
int     smtp_mode;
int     smtp_host_lookup_mask;
int     smtp_dns_support;
STRING_LIST *smtp_cache_dest;
SCACHE *smtp_scache;
MAPS   *smtp_ehlo_dis_maps;
MAPS   *smtp_generic_maps;
int     smtp_ext_prop_mask;
unsigned smtp_dns_res_opt;
MAPS   *smtp_pix_bug_maps;
HBC_CHECKS *smtp_header_checks;		/* limited header checks */
HBC_CHECKS *smtp_body_checks;		/* limited body checks */

#ifdef USE_TLS

 /*
  * OpenSSL client state (opaque handle)
  */
TLS_APPL_STATE *smtp_tls_ctx;
int     smtp_tls_insecure_mx_policy;

#endif

 /*
  * IPv6 preference.
  */
static int smtp_addr_pref;

/* deliver_message - deliver message with extreme prejudice */

static int deliver_message(const char *service, DELIVER_REQUEST *request)
{
    SMTP_STATE *state;
    int     result;

    if (msg_verbose)
	msg_info("deliver_message: from %s", request->sender);

    /*
     * Sanity checks. The smtp server is unprivileged and chrooted, so we can
     * afford to distribute the data censoring code, instead of having it all
     * in one place.
     */
    if (request->nexthop[0] == 0)
	msg_fatal("empty nexthop hostname");
    if (request->rcpt_list.len <= 0)
	msg_fatal("recipient count: %d", request->rcpt_list.len);

    /*
     * Initialize. Bundle all information about the delivery request, so that
     * we can produce understandable diagnostics when something goes wrong
     * many levels below. The alternative would be to make everything global.
     */
    state = smtp_state_alloc();
    state->request = request;
    state->src = request->fp;
    state->service = service;
    state->misc_flags |= smtp_addr_pref;
    SMTP_RCPT_INIT(state);

    /*
     * Establish an SMTP session and deliver this message to all requested
     * recipients. At the end, notify the postmaster of any protocol errors.
     * Optionally deliver mail locally when this machine is the best mail
     * exchanger.
     */
    result = smtp_connect(state);

    /*
     * Clean up.
     */
    smtp_state_free(state);

    return (result);
}

/* smtp_service - perform service for client */

static void smtp_service(VSTREAM *client_stream, char *service, char **argv)
{
    DELIVER_REQUEST *request;
    int     status;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to remote SMTP delivery service. What we see below is a
     * little protocol to (1) tell the queue manager that we are ready, (2)
     * read a request from the queue manager, and (3) report the completion
     * status of that request. All connection-management stuff is handled by
     * the common code in single_server.c.
     */
    if ((request = deliver_request_read(client_stream)) != 0) {
	status = deliver_message(service, request);
	deliver_request_done(client_stream, request, status);
    }
}

/* post_init - post-jail initialization */

static void post_init(char *unused_name, char **unused_argv)
{
    static const NAME_MASK lookup_masks[] = {
	SMTP_HOST_LOOKUP_DNS, SMTP_HOST_FLAG_DNS,
	SMTP_HOST_LOOKUP_NATIVE, SMTP_HOST_FLAG_NATIVE,
	0,
    };
    static const NAME_MASK dns_res_opt_masks[] = {
	SMTP_DNS_RES_OPT_DEFNAMES, RES_DEFNAMES,
	SMTP_DNS_RES_OPT_DNSRCH, RES_DNSRCH,
	0,
    };
    static const NAME_CODE dns_support[] = {
	SMTP_DNS_SUPPORT_DISABLED, SMTP_DNS_DISABLED,
	SMTP_DNS_SUPPORT_ENABLED, SMTP_DNS_ENABLED,
#if (RES_USE_DNSSEC != 0) && (RES_USE_EDNS0 != 0)
	SMTP_DNS_SUPPORT_DNSSEC, SMTP_DNS_DNSSEC,
#endif
	0, SMTP_DNS_INVALID,
    };

    if (*var_smtp_dns_support == 0) {
	/* Backwards compatible empty setting */
	smtp_dns_support =
	    var_disable_dns ? SMTP_DNS_DISABLED : SMTP_DNS_ENABLED;
    } else {
	smtp_dns_support =
	    name_code(dns_support, NAME_CODE_FLAG_NONE, var_smtp_dns_support);
	if (smtp_dns_support == SMTP_DNS_INVALID)
	    msg_fatal("invalid %s: \"%s\"", VAR_LMTP_SMTP(DNS_SUPPORT),
		      var_smtp_dns_support);
	var_disable_dns = (smtp_dns_support == SMTP_DNS_DISABLED);
    }

#ifdef USE_TLS
    if (smtp_mode) {
	smtp_tls_insecure_mx_policy =
	    tls_level_lookup(var_smtp_tls_insecure_mx_policy);
	switch (smtp_tls_insecure_mx_policy) {
	case TLS_LEV_MAY:
	case TLS_LEV_ENCRYPT:
	case TLS_LEV_DANE:
	    break;
	default:
	    msg_fatal("invalid %s: \"%s\"", VAR_SMTP_TLS_INSECURE_MX_POLICY,
		      var_smtp_tls_insecure_mx_policy);
	}
    }
#endif

    /*
     * Select hostname lookup mechanisms.
     */
    if (smtp_dns_support == SMTP_DNS_DISABLED)
	smtp_host_lookup_mask = SMTP_HOST_FLAG_NATIVE;
    else
	smtp_host_lookup_mask =
	    name_mask(VAR_LMTP_SMTP(HOST_LOOKUP), lookup_masks,
		      var_smtp_host_lookup);
    if (msg_verbose)
	msg_info("host name lookup methods: %s",
		 str_name_mask(VAR_LMTP_SMTP(HOST_LOOKUP), lookup_masks,
			       smtp_host_lookup_mask));

    /*
     * Session cache instance.
     */
    if (*var_smtp_cache_dest || var_smtp_cache_demand)
#if 0
	smtp_scache = scache_multi_create();
#else
	smtp_scache = scache_clnt_create(var_scache_service,
					 var_scache_proto_tmout,
					 var_ipc_idle_limit,
					 var_ipc_ttl_limit);
#endif

    /*
     * Select DNS query flags.
     */
    smtp_dns_res_opt = name_mask(VAR_LMTP_SMTP(DNS_RES_OPT), dns_res_opt_masks,
				 var_smtp_dns_res_opt);

    /*
     * Address verification.
     */
    smtp_vrfy_init();
}

/* pre_init - pre-jail initialization */

static void pre_init(char *unused_name, char **unused_argv)
{
    int     use_tls;
    static const NAME_CODE addr_pref_map[] = {
	INET_PROTO_NAME_IPV6, SMTP_MISC_FLAG_PREF_IPV6,
	INET_PROTO_NAME_IPV4, SMTP_MISC_FLAG_PREF_IPV4,
	INET_PROTO_NAME_ANY, 0,
	0, -1,
    };

    /*
     * Turn on per-peer debugging.
     */
    debug_peer_init();

    /*
     * SASL initialization.
     */
    if (var_smtp_sasl_enable)
#ifdef USE_SASL_AUTH
	smtp_sasl_initialize();
#else
	msg_warn("%s is true, but SASL support is not compiled in",
		 VAR_LMTP_SMTP(SASL_ENABLE));
#endif

    if (*var_smtp_tls_level != 0)
	switch (tls_level_lookup(var_smtp_tls_level)) {
	case TLS_LEV_SECURE:
	case TLS_LEV_VERIFY:
	case TLS_LEV_DANE_ONLY:
	case TLS_LEV_FPRINT:
	case TLS_LEV_ENCRYPT:
	    var_smtp_use_tls = var_smtp_enforce_tls = 1;
	    break;
	case TLS_LEV_DANE:
	case TLS_LEV_MAY:
	    var_smtp_use_tls = 1;
	    var_smtp_enforce_tls = 0;
	    break;
	case TLS_LEV_NONE:
	    var_smtp_use_tls = var_smtp_enforce_tls = 0;
	    break;
	default:
	    /* tls_level_lookup() logs no warning. */
	    /* session_tls_init() assumes that var_smtp_tls_level is sane. */
	    msg_fatal("Invalid TLS level \"%s\"", var_smtp_tls_level);
	}
    use_tls = (var_smtp_use_tls || var_smtp_enforce_tls);

    /*
     * Initialize the TLS data before entering the chroot jail
     */
    if (use_tls || var_smtp_tls_per_site[0] || var_smtp_tls_policy[0]) {
#ifdef USE_TLS
	TLS_CLIENT_INIT_PROPS props;

	/*
	 * We get stronger type safety and a cleaner interface by combining
	 * the various parameters into a single tls_client_props structure.
	 * 
	 * Large parameter lists are error-prone, so we emulate a language
	 * feature that C does not have natively: named parameter lists.
	 */
	smtp_tls_ctx =
	    TLS_CLIENT_INIT(&props,
			    log_param = VAR_LMTP_SMTP(TLS_LOGLEVEL),
			    log_level = var_smtp_tls_loglevel,
			    verifydepth = var_smtp_tls_scert_vd,
			    cache_type = LMTP_SMTP_SUFFIX(TLS_MGR_SCACHE),
			    cert_file = var_smtp_tls_cert_file,
			    key_file = var_smtp_tls_key_file,
			    dcert_file = var_smtp_tls_dcert_file,
			    dkey_file = var_smtp_tls_dkey_file,
			    eccert_file = var_smtp_tls_eccert_file,
			    eckey_file = var_smtp_tls_eckey_file,
			    CAfile = var_smtp_tls_CAfile,
			    CApath = var_smtp_tls_CApath,
			    mdalg = var_smtp_tls_fpt_dgst);
	smtp_tls_list_init();
#else
	msg_warn("TLS has been selected, but TLS support is not compiled in");
#endif
    }

    /*
     * Flush client.
     */
    flush_init();

    /*
     * Session cache domain list.
     */
    if (*var_smtp_cache_dest)
	smtp_cache_dest = string_list_init(VAR_SMTP_CACHE_DEST,
					   MATCH_FLAG_RETURN,
					   var_smtp_cache_dest);

    /*
     * EHLO keyword filter.
     */
    if (*var_smtp_ehlo_dis_maps)
	smtp_ehlo_dis_maps = maps_create(VAR_LMTP_SMTP(EHLO_DIS_MAPS),
					 var_smtp_ehlo_dis_maps,
					 DICT_FLAG_LOCK);

    /*
     * PIX bug workarounds.
     */
    if (*var_smtp_pix_bug_maps)
	smtp_pix_bug_maps = maps_create(VAR_LMTP_SMTP(PIX_BUG_MAPS),
					var_smtp_pix_bug_maps,
					DICT_FLAG_LOCK);

    /*
     * Generic maps.
     */
    if (*var_prop_extension)
	smtp_ext_prop_mask =
	    ext_prop_mask(VAR_PROP_EXTENSION, var_prop_extension);
    if (*var_smtp_generic_maps)
	smtp_generic_maps =
	    maps_create(VAR_LMTP_SMTP(GENERIC_MAPS), var_smtp_generic_maps,
			DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
			| DICT_FLAG_UTF8_REQUEST);

    /*
     * Header/body checks.
     */
    smtp_header_checks = hbc_header_checks_create(
			       VAR_LMTP_SMTP(HEAD_CHKS), var_smtp_head_chks,
			       VAR_LMTP_SMTP(MIME_CHKS), var_smtp_mime_chks,
			       VAR_LMTP_SMTP(NEST_CHKS), var_smtp_nest_chks,
						  smtp_hbc_callbacks);
    smtp_body_checks = hbc_body_checks_create(
			       VAR_LMTP_SMTP(BODY_CHKS), var_smtp_body_chks,
					      smtp_hbc_callbacks);

    /*
     * Server reply filter.
     */
    if (*var_smtp_resp_filter)
	smtp_chat_resp_filter =
	    dict_open(var_smtp_resp_filter, O_RDONLY,
		      DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX);

    /*
     * Address family preference.
     */
    if (*var_smtp_addr_pref) {
	smtp_addr_pref = name_code(addr_pref_map, NAME_CODE_FLAG_NONE,
				   var_smtp_addr_pref);
	if (smtp_addr_pref < 0)
	    msg_fatal("bad %s value: %s", VAR_LMTP_SMTP(ADDR_PREF),
		      var_smtp_addr_pref);
    }

    /*
     * DNS reply filter.
     */
    if (*var_smtp_dns_re_filter)
	dns_rr_filter_compile(VAR_LMTP_SMTP(DNS_RE_FILTER),
			      var_smtp_dns_re_filter);
}

/* pre_accept - see if tables have changed */

static void pre_accept(char *unused_name, char **unused_argv)
{
    const char *table;

    if ((table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the single-threaded skeleton */

int     main(int argc, char **argv)
{
    char   *sane_procname;

#include "smtp_params.c"
#include "lmtp_params.c"

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * XXX At this point, var_procname etc. are not initialized.
     * 
     * The process name, "smtp" or "lmtp", determines the protocol, the DSN
     * server reply type, SASL service information lookup, and more. Prepare
     * for the possibility there may be another personality.
     */
    sane_procname = sane_basename((VSTRING *) 0, argv[0]);
    if (strcmp(sane_procname, "smtp") == 0)
	smtp_mode = 1;
    else if (strcmp(sane_procname, "lmtp") == 0)
	smtp_mode = 0;
    else
	msg_fatal("unexpected process name \"%s\" - "
		  "specify \"smtp\" or \"lmtp\"", var_procname);

    /*
     * Initialize with the LMTP or SMTP parameter name space.
     */
    single_server_main(argc, argv, smtp_service,
		       CA_MAIL_SERVER_TIME_TABLE(smtp_mode ?
					 smtp_time_table : lmtp_time_table),
		       CA_MAIL_SERVER_INT_TABLE(smtp_mode ?
					   smtp_int_table : lmtp_int_table),
		       CA_MAIL_SERVER_STR_TABLE(smtp_mode ?
					   smtp_str_table : lmtp_str_table),
		       CA_MAIL_SERVER_BOOL_TABLE(smtp_mode ?
					 smtp_bool_table : lmtp_bool_table),
		       CA_MAIL_SERVER_PRE_INIT(pre_init),
		       CA_MAIL_SERVER_POST_INIT(post_init),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_BOUNCE_INIT(VAR_SMTP_DSN_FILTER,
						  &var_smtp_dsn_filter),
		       0);
}
