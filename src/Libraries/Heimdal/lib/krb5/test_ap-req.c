/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include <config.h>

#include <sys/types.h>
#include <stdio.h>
#include <krb5.h>
#include <err.h>
#include <getarg.h>
#include <roken.h>

static int verify_pac = 0;
static int server_any = 0;
static int version_flag = 0;
static int help_flag	= 0;

static struct getargs args[] = {
    {"verify-pac",0,	arg_flag,	&verify_pac,
     "verify the PAC", NULL },
    {"server-any",0,	arg_flag,	&server_any,
     "let server pick the principal", NULL },
    {"version",	0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args, sizeof(args)/sizeof(*args), NULL, "...");
    exit (ret);
}


static void
test_ap(krb5_context context,
	krb5_principal target,
	krb5_principal server,
	krb5_keytab keytab,
	krb5_ccache ccache,
	const krb5_flags client_flags)
{
    krb5_error_code ret;
    krb5_auth_context client_ac = NULL, server_ac = NULL;
    krb5_data client_request;
    krb5_data data;
    krb5_flags server_flags;
    krb5_ticket *ticket = NULL;
    int32_t server_seq, client_seq;
    krb5_rcache rcache = NULL;

    krb5_data_zero(&data);
    krb5_data_zero(&client_request);

    ret = krb5_mk_req_exact(context,
			    &client_ac,
			    client_flags,
			    target,
			    NULL,
			    ccache,
			    &client_request);
    if (ret)
	krb5_err(context, 1, ret, "krb5_mk_req_exact");

    ret = krb5_rd_req(context,
		      &server_ac,
		      &client_request,
		      server,
		      keytab,
		      &server_flags,
		      &ticket);
    if (ret)
	krb5_err(context, 1, ret, "krb5_rd_req");


    if (server_flags & AP_OPTS_MUTUAL_REQUIRED) {
	krb5_ap_rep_enc_part *repl;

	if ((client_flags & AP_OPTS_MUTUAL_REQUIRED) == 0)
	    krb5_errx(context, 1, "client flag missing mutual req");

	ret = krb5_mk_rep (context, server_ac, &data);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_mk_rep");

	ret = krb5_rd_rep (context,
			   client_ac,
			   &data,
			   &repl);
	if (ret)
	    krb5_err(context, 1, ret, "krb5_rd_rep");

	krb5_data_free(&data);
	krb5_free_ap_rep_enc_part (context, repl);
    } else {
	if (client_flags & AP_OPTS_MUTUAL_REQUIRED)
	    krb5_errx(context, 1, "server flag missing mutual req");
    }

    krb5_auth_con_getremoteseqnumber(context, server_ac, &server_seq);
    krb5_auth_con_getremoteseqnumber(context, client_ac, &client_seq);
    if (server_seq != client_seq)
	krb5_errx(context, 1, "seq num differ");

    krb5_auth_con_getlocalseqnumber(context, server_ac, &server_seq);
    krb5_auth_con_getlocalseqnumber(context, client_ac, &client_seq);
    if (server_seq != client_seq)
	krb5_errx(context, 1, "seq num differ");

    krb5_auth_con_free(context, client_ac);
    client_ac = NULL;

    krb5_auth_con_free(context, server_ac);
    server_ac = NULL;

    if (verify_pac) {
	krb5_pac pac;

	ret = krb5_ticket_get_authorization_data_type(context,
						      ticket,
						      KRB5_AUTHDATA_WIN2K_PAC,
						      &data);
	if (ret)
	    krb5_err(context, 1, ret, "get pac");

	ret = krb5_pac_parse(context, data.data, data.length, &pac);
	if (ret)
	    krb5_err(context, 1, ret, "pac parse");

	krb5_pac_free(context, pac);
	krb5_data_free(&data);
    }

    krb5_free_ticket(context, ticket);

    /*
     * Check replays
     */

    ret = krb5_get_server_rcache(context, NULL, &rcache);
    if (ret)
	krb5_err(context, 1, ret, "krb5_get_server_rcache");

    krb5_auth_con_init(context, &server_ac);
    krb5_auth_con_setrcache(context, server_ac, rcache);

    ret = krb5_rd_req(context,
		      &server_ac,
		      &client_request,
		      server,
		      keytab,
		      &server_flags,
		      &ticket);
    if (ret)
	krb5_err(context, 1, ret, "krb5_rd_req");

    krb5_auth_con_free(context, server_ac);
    server_ac = NULL;

    krb5_auth_con_init(context, &server_ac);
    krb5_auth_con_setrcache(context, server_ac, rcache);

    ret = krb5_rd_req(context,
		      &server_ac,
		      &client_request,
		      server,
		      keytab,
		      &server_flags,
		      &ticket);
    if (ret != KRB5_RC_REPLAY)
	krb5_err(context, 1, ret, "krb5_rd_req not detecting replays");

    krb5_auth_con_free(context, server_ac);
    server_ac = NULL;

    krb5_data_free(&client_request);
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    int optidx = 0;
    const char *principal, *keytab, *ccache;
    krb5_ccache id;
    krb5_keytab kt;
    krb5_principal sprincipal, server;

    setprogname(argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    argc -= optidx;
    argv += optidx;

    if (argc < 3)
	usage(1);

    principal = argv[0];
    keytab = argv[1];
    ccache = argv[2];

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = krb5_cc_resolve(context, ccache, &id);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_resolve");

    ret = krb5_parse_name(context, principal, &sprincipal);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_kt_resolve(context, keytab, &kt);
    if (ret)
	krb5_err(context, 1, ret, "krb5_kt_resolve");

    if (server_any)
	server = NULL;
    else
	server = sprincipal;

    test_ap(context, sprincipal, server, kt, id, 0);
    test_ap(context, sprincipal, server, kt, id, AP_OPTS_MUTUAL_REQUIRED);

    krb5_cc_close(context, id);
    krb5_kt_close(context, kt);
    krb5_free_principal(context, sprincipal);

    krb5_free_context(context);

    return ret;
}
