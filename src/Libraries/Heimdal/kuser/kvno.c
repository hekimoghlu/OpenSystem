/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "kuser_locl.h"

static char *etype_str = NULL;
static char *ccache_name = NULL;
static char *keytab_name = NULL;
static char *sname = NULL;

static int  version_flag = 0;
static int  help_flag = 0;
static int  quiet_flag = 0;

static void do_v5_kvno (int argc, char *argv[],
                        char *ccache_name, char *etype_str, char *keytab_name,
			char *sname);

struct getargs args[] = {
    { "enctype",        'e', arg_string, &etype_str,
      NP_("Encryption type to use", ""), "enctype" },
    { "cache",          'c', arg_string, &ccache_name,
      NP_("Credentials cache", ""), "cachename" },
    { "keytab",         'k', arg_string, &keytab_name,
      NP_("Keytab to use", ""), "keytabname" },
    { "server",         'S', arg_string, &sname,
      NP_("Server to get ticket for", ""), "principal" },
    { "quiet",          'q', arg_flag, &quiet_flag,
      NP_("Quiet", "") },
    { "version",        0, arg_flag, &version_flag },
    { "help",           0, arg_flag, &help_flag }
};

static void
usage(int ret)
{
    arg_printusage_i18n (args, sizeof(args)/sizeof(*args),
                         N_("Usage: ", ""), NULL,
                         "principal1 [principal2 ...]",
                         getarg_i18n);
    exit (ret);
}

int main(int argc, char *argv[])
{
    int optidx = 0;

    setprogname (argv[0]);

    setlocale(LC_ALL, "");
    bindtextdomain ("heimdal_kuser", HEIMDAL_LOCALEDIR);
    textdomain("heimdal_kuser");

    if (getarg(args, sizeof(args)/sizeof(args[0]), argc, argv, &optidx))
        usage(1);

    if (help_flag)
        usage (0);

    if (version_flag) {
        print_version(NULL);
        exit (0);
    }

    argc -= optidx;
    argv += optidx;

    do_v5_kvno(argc, argv, ccache_name, etype_str, keytab_name, sname);

    return 0;
}

static void do_v5_kvno (int count, char *names[],
                        char * ccache_name, char *etype_str, char *keytab_name,
			char *sname)
{
    krb5_error_code ret;
    krb5_context context = 0;
    int i, errors;
    krb5_enctype etype;
    krb5_ccache ccache;
    krb5_principal me;
    krb5_creds in_creds, *out_creds = NULL;
    Ticket ticket;
    size_t len;
    char *princ = NULL;
    krb5_keytab keytab = NULL;

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context failed: %d", ret);

    if (etype_str) {
        ret = krb5_string_to_enctype(context, etype_str, &etype);
	if (ret)
	    krb5_err(context, 1, ret, "Failed to convert encryption type %s", etype_str);
    } else {
	etype = 0;
    }

    if (ccache_name)
        ret = krb5_cc_resolve(context, ccache_name, &ccache);
    else
        ret = krb5_cc_default(context, &ccache);
    if (ret)
        krb5_err(context, 1, ret, "Failed to open credentials cache %s",
                 (ccache_name) ? ccache_name : "(Default)");

    if (keytab_name) {
	ret = krb5_kt_resolve(context, keytab_name, &keytab);
	if (ret)
            krb5_err(context, 1, ret, "Can't resolve keytab %s", keytab_name);
    }

    ret = krb5_cc_get_principal(context, ccache, &me);
    if (ret)
        krb5_err(context, 1, ret, "krb5_cc_get_principal");

    errors = 0;

    for (i = 0; i < count; i++) {
	memset(&in_creds, 0, sizeof(in_creds));
        memset(&ticket, 0, sizeof(ticket));

	in_creds.client = me;

	if (sname != NULL) {
	    ret = krb5_sname_to_principal(context, names[i],
					  sname, KRB5_NT_SRV_HST,
					  &in_creds.server);
	} else {
	    ret = krb5_parse_name(context, names[i], &in_creds.server);
	}
	if (ret) {
	    if (!quiet_flag)
                krb5_warn(context, ret, "Couldn't parse principal name %s", names[i]);
            errors++;
	    continue;
	}

	ret = krb5_unparse_name(context, in_creds.server, &princ);
	if (ret) {
            krb5_warn(context, ret, "Couldn't format parsed principal name for '%s'",
                      names[i]);
	    errors++;
            goto next;
	}

	in_creds.session.keytype = etype;

	ret = krb5_get_credentials(context, 0, ccache, &in_creds, &out_creds);

	if (ret) {
            krb5_warn(context, ret, "Couldn't get credentials for %s", princ);
	    errors++;
	    goto next;
	}

	ret = decode_Ticket(out_creds->ticket.data, out_creds->ticket.length,
                            &ticket, &len);
	if (ret) {
	    krb5_err(context, 1, ret, "Can't decode ticket for %s", princ);
	    errors++;
            goto next;
	    continue;
	}

	if (keytab) {
            krb5_keytab_entry   kte;
            krb5_crypto         crypto;
            krb5_data           dec_data;
            EncTicketPart       decr_part;

            ret = krb5_kt_get_entry(context, keytab, in_creds.server,
                                    (ticket.enc_part.kvno != NULL)?
                                    *ticket.enc_part.kvno : 0,
                                    ticket.enc_part.etype,
                                    &kte);
            if (ret) {
                krb5_warn(context, ret, "Can't decrypt ticket for %s", princ);
                if (!quiet_flag)
                    printf("%s: kvno = %d, keytab entry invalid", princ,
                           (ticket.enc_part.kvno != NULL)?
                           *ticket.enc_part.kvno : 0);
                errors ++;
                goto next;
            }

            ret = krb5_crypto_init(context, &kte.keyblock, 0, &crypto);
            if (ret) {
                krb5_warn(context, ret, "krb5_crypto_init");
                errors ++;
                krb5_kt_free_entry(context, &kte);
                goto next;
            }

            ret = krb5_decrypt_EncryptedData (context, crypto, KRB5_KU_TICKET,
                                              &ticket.enc_part, &dec_data);
            krb5_crypto_destroy(context, crypto);
            krb5_kt_free_entry(context, &kte);

            if (ret) {
                krb5_warn(context, ret, "krb5_decrypt_EncryptedData");
                errors ++;
                goto next;
            }

            ret = decode_EncTicketPart(dec_data.data, dec_data.length,
                                       &decr_part, &len);
            krb5_data_free(&dec_data);
            if (ret) {
                krb5_warn(context, ret, "decode_EncTicketPart");
                errors ++;
                goto next;
            }

            if (!quiet_flag)
		printf("%s: kvno = %d, keytab entry valid\n", princ,
                       (ticket.enc_part.kvno != NULL)?
                       *ticket.enc_part.kvno : 0);

            free_EncTicketPart(&decr_part);
	} else {
	    if (!quiet_flag)
		printf("%s: kvno = %d\n", princ,
                       (ticket.enc_part.kvno != NULL)? *ticket.enc_part.kvno : 0);
	}

    next:
        if (out_creds) {
            krb5_free_creds(context, out_creds);
            out_creds = NULL;
        }

        if (princ) {
            krb5_free_unparsed_name(context, princ);
            princ = NULL;
        }

	krb5_free_principal(context, in_creds.server);

        free_Ticket(&ticket);
    }

    if (keytab)
	krb5_kt_close(context, keytab);
    krb5_free_principal(context, me);
    krb5_cc_close(context, ccache);
    krb5_free_context(context);

    if (errors)
	exit(1);

    exit(0);
}
