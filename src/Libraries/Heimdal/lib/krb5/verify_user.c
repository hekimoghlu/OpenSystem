/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "krb5_locl.h"

static krb5_error_code
verify_common (krb5_context context,
	       krb5_principal principal,
	       krb5_principal server_principal,
	       krb5_ccache ccache,
	       krb5_keytab keytab,
	       krb5_boolean secure,
	       const char *service,
	       krb5_creds *cred)
{
    krb5_error_code ret;
    krb5_verify_init_creds_opt vopt;
    krb5_ccache id;

    krb5_verify_init_creds_opt_init(&vopt);
    krb5_verify_init_creds_opt_set_ap_req_nofail(&vopt, secure);
    krb5_verify_init_creds_opt_set_service(&vopt, service);

    ret = krb5_verify_init_creds(context,
				 cred,
				 server_principal,
				 keytab,
				 NULL,
				 &vopt);
    if(ret)
	return ret;
    if(ccache == NULL)
	ret = krb5_cc_default (context, &id);
    else
	id = ccache;
    if(ret == 0){
	ret = krb5_cc_initialize(context, id, principal);
	if(ret == 0){
	    ret = krb5_cc_store_cred(context, id, cred);
	}
	if(ccache == NULL)
	    krb5_cc_close(context, id);
    }
    return ret;
}

struct krb5_verify_opt {
    unsigned int flags;
    krb5_ccache ccache;
    krb5_keytab keytab;
    krb5_boolean secure;
    const char *service;
    krb5_principal server;
    krb5_prompter_fct prompter;
    void *prompter_data;
};

/*
 * Verify user `principal' with `password'.
 *
 * If `secure', also verify against local service key for `service'.
 *
 * As a side effect, fresh tickets are obtained and stored in `ccache'.
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_init(krb5_verify_opt *opt)
{
    memset(opt, 0, sizeof(*opt));
    opt->secure = TRUE;
    opt->service = "host";
    opt->prompter = krb5_prompter_posix;
}

KRB5_LIB_FUNCTION int KRB5_LIB_CALL
krb5_verify_opt_alloc(krb5_context context, krb5_verify_opt **opt)
{
    *opt = calloc(1, sizeof(**opt));
    if ((*opt) == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    krb5_verify_opt_init(*opt);
    return 0;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_free(krb5_verify_opt *opt)
{
    free(opt);
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_ccache(krb5_verify_opt *opt, krb5_ccache ccache)
{
    opt->ccache = ccache;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_server(krb5_verify_opt *opt, krb5_principal server)
{
    opt->server = server;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_keytab(krb5_verify_opt *opt, krb5_keytab keytab)
{
    opt->keytab = keytab;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_secure(krb5_verify_opt *opt, krb5_boolean secure)
{
    opt->secure = secure;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_service(krb5_verify_opt *opt, const char *service)
{
    opt->service = service;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_flags(krb5_verify_opt *opt, unsigned int flags)
{
    opt->flags |= flags;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_verify_opt_set_prompter(krb5_verify_opt *opt,
			     krb5_prompter_fct prompter,
			     void *prompter_data)
{
    opt->prompter = prompter;
    opt->prompter_data = prompter_data;
}


static krb5_error_code
verify_user_opt_int(krb5_context context,
		    krb5_principal principal,
		    const char *password,
		    krb5_verify_opt *vopt)

{
    krb5_error_code ret;
    krb5_get_init_creds_opt *opt;
    krb5_creds cred;

    memset(&cred, 0, sizeof(cred));

    ret = krb5_get_init_creds_opt_alloc(context, &opt);
    if (ret)
	return ret;
    krb5_get_init_creds_opt_set_default_flags(context, NULL,
					      krb5_principal_get_realm(context, principal),
					      opt);

    ret = krb5_get_init_creds_password(context,
				       &cred,
				       principal,
				       password,
				       vopt->prompter,
				       vopt->prompter_data,
				       0,
				       NULL,
				       opt);
    if (ret) {
	krb5_get_init_creds_opt_free(context, opt);
	return ret;
    }
#define OPT(V, D) ((vopt && (vopt->V)) ? (vopt->V) : (D))
    ret = verify_common(context, principal, OPT(server, NULL),
			OPT(ccache, NULL),
			OPT(keytab, NULL),
			vopt ? vopt->secure : TRUE,
			OPT(service, "host"), &cred);
#undef OPT
    krb5_free_cred_contents(context, &cred);
    krb5_get_init_creds_opt_free(context, opt);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_verify_user_opt(krb5_context context,
		     krb5_principal principal,
		     const char *password,
		     krb5_verify_opt *opt)
{
    krb5_error_code ret;

    if(opt && (opt->flags & KRB5_VERIFY_LREALMS)) {
	krb5_realm *realms, *r;
	ret = krb5_get_default_realms (context, &realms);
	if (ret)
	    return ret;
	ret = KRB5_CONFIG_NODEFREALM;

	for (r = realms; *r != NULL && ret != 0; ++r) {
	    ret = krb5_principal_set_realm(context, principal, *r);
	    if (ret) {
		krb5_free_host_realm (context, realms);
		return ret;
	    }

	    ret = verify_user_opt_int(context, principal, password, opt);
	}
	krb5_free_host_realm (context, realms);
	if(ret)
	    return ret;
    } else
	ret = verify_user_opt_int(context, principal, password, opt);
    return ret;
}

/* compat function that calls above */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_verify_user(krb5_context context,
		 krb5_principal principal,
		 krb5_ccache ccache,
		 const char *password,
		 krb5_boolean secure,
		 const char *service)
{
    krb5_verify_opt opt;

    krb5_verify_opt_init(&opt);

    krb5_verify_opt_set_ccache(&opt, ccache);
    krb5_verify_opt_set_secure(&opt, secure);
    krb5_verify_opt_set_service(&opt, service);

    return krb5_verify_user_opt(context, principal, password, &opt);
}

/*
 * A variant of `krb5_verify_user'.  The realm of `principal' is
 * ignored and all the local realms are tried.
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_verify_user_lrealm(krb5_context context,
			krb5_principal principal,
			krb5_ccache ccache,
			const char *password,
			krb5_boolean secure,
			const char *service)
{
    krb5_verify_opt opt;

    krb5_verify_opt_init(&opt);

    krb5_verify_opt_set_ccache(&opt, ccache);
    krb5_verify_opt_set_secure(&opt, secure);
    krb5_verify_opt_set_service(&opt, service);
    krb5_verify_opt_set_flags(&opt, KRB5_VERIFY_LREALMS);

    return krb5_verify_user_opt(context, principal, password, &opt);
}
