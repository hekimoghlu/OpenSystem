/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

/**
 * @page krb5_init_creds_intro The initial credential handing functions
 * @section section_krb5_init_creds Initial credential
 *
 * Functions to get initial credentials: @ref krb5_credential .
 */

/**
 * Allocate a new krb5_get_init_creds_opt structure, free with
 * krb5_get_init_creds_opt_free().
 *
 * @ingroup krb5_credential
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_alloc(krb5_context context,
			      krb5_get_init_creds_opt **opt)
{
    krb5_get_init_creds_opt *o;

    *opt = NULL;
    o = calloc(1, sizeof(*o));
    if (o == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    o->opt_private = calloc(1, sizeof(*o->opt_private));
    if (o->opt_private == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	free(o);
	return ENOMEM;
    }
    o->opt_private->refcount = 1;
    *opt = o;
    return 0;
}

/**
 * Free krb5_get_init_creds_opt structure.
 *
 * @ingroup krb5_credential
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_free(krb5_context context,
			     krb5_get_init_creds_opt *opt)
{
    if (opt == NULL || opt->opt_private == NULL)
	return;
    if (opt->opt_private->refcount < 1) /* abort ? */
	return;
    if (--opt->opt_private->refcount == 0) {
	_krb5_get_init_creds_opt_free_pkinit(opt);
	free(opt->opt_private);
    }
    memset(opt, 0, sizeof(*opt));
    free(opt);
}

static krb5_deltat
get_config_time (krb5_context context,
		 const char *realm,
		 const char *name,
		 int def)
{
    krb5_deltat ret;

    ret = krb5_config_get_time (context, NULL,
				"realms",
				realm,
				name,
				NULL);
    if (ret >= 0)
	return ret;
    ret = krb5_config_get_time (context, NULL,
				"libdefaults",
				name,
				NULL);
    if (ret >= 0)
	return ret;
    return def;
}

static krb5_boolean
get_config_bool (krb5_context context,
		 krb5_boolean def_value,
		 const char *realm,
		 const char *name)
{
    krb5_boolean b;

    b = krb5_config_get_bool_default(context, NULL, def_value,
				     "realms", realm, name, NULL);
    if (b != def_value)
	return b;
    b = krb5_config_get_bool_default (context, NULL, def_value,
				      "libdefaults", name, NULL);
    if (b != def_value)
	return b;
    return def_value;
}

/*
 * set all the values in `opt' to the appropriate values for
 * application `appname' (default to getprogname() if NULL), and realm
 * `realm'.  First looks in [appdefaults] but falls back to
 * [realms] or [libdefaults] for some of the values.
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_default_flags(krb5_context context,
					  const char *appname,
					  krb5_const_realm realm,
					  krb5_get_init_creds_opt *opt)
{
    krb5_boolean b;
    time_t t;

    b = get_config_bool (context, KRB5_FORWARDABLE_DEFAULT,
			 realm, "forwardable");
    krb5_appdefault_boolean(context, appname, realm, "forwardable", b, &b);
    krb5_get_init_creds_opt_set_forwardable(opt, b);

    b = get_config_bool (context, FALSE, realm, "proxiable");
    krb5_appdefault_boolean(context, appname, realm, "proxiable", b, &b);
    krb5_get_init_creds_opt_set_proxiable (opt, b);

    krb5_appdefault_time(context, appname, realm, "ticket_lifetime", 0, &t);
    if (t == 0)
	t = get_config_time (context, realm, "ticket_lifetime", 0);
    if(t != 0)
	krb5_get_init_creds_opt_set_tkt_life(opt, t);

    krb5_appdefault_time(context, appname, realm, "renew_lifetime", 0, &t);
    if (t == 0)
	t = get_config_time (context, realm, "renew_lifetime", 0);
    if(t != 0)
	krb5_get_init_creds_opt_set_renew_life(opt, t);

    krb5_appdefault_boolean(context, appname, realm, "no-addresses",
			    KRB5_ADDRESSLESS_DEFAULT, &b);
    krb5_get_init_creds_opt_set_addressless (context, opt, b);

#if 0
    krb5_appdefault_boolean(context, appname, realm, "anonymous", FALSE, &b);
    krb5_get_init_creds_opt_set_anonymous (opt, b);

    krb5_get_init_creds_opt_set_etype_list(opt, enctype,
					   etype_str.num_strings);

    krb5_get_init_creds_opt_set_salt(krb5_get_init_creds_opt *opt,
				     krb5_data *salt);

    krb5_get_init_creds_opt_set_preauth_list(krb5_get_init_creds_opt *opt,
					     krb5_preauthtype *preauth_list,
					     int preauth_list_length);
#endif
}


KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_tkt_life(krb5_get_init_creds_opt *opt,
				     krb5_deltat tkt_life)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_TKT_LIFE;
    opt->tkt_life = tkt_life;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_renew_life(krb5_get_init_creds_opt *opt,
				       krb5_deltat renew_life)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_RENEW_LIFE;
    opt->renew_life = renew_life;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_forwardable(krb5_get_init_creds_opt *opt,
					int forwardable)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_FORWARDABLE;
    opt->forwardable = forwardable;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_proxiable(krb5_get_init_creds_opt *opt,
				      int proxiable)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_PROXIABLE;
    opt->proxiable = proxiable;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_etype_list(krb5_get_init_creds_opt *opt,
				       krb5_enctype *etype_list,
				       int etype_list_length)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_ETYPE_LIST;
    opt->etype_list = etype_list;
    opt->etype_list_length = etype_list_length;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_address_list(krb5_get_init_creds_opt *opt,
					 krb5_addresses *addresses)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_ADDRESS_LIST;
    opt->address_list = addresses;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_preauth_list(krb5_get_init_creds_opt *opt,
					 krb5_preauthtype *preauth_list,
					 int preauth_list_length)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_PREAUTH_LIST;
    opt->preauth_list_length = preauth_list_length;
    opt->preauth_list = preauth_list;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_salt(krb5_get_init_creds_opt *opt,
				 krb5_data *salt)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_SALT;
    opt->salt = salt;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_set_anonymous(krb5_get_init_creds_opt *opt,
				      int anonymous)
{
    opt->flags |= KRB5_GET_INIT_CREDS_OPT_ANONYMOUS;
    opt->anonymous = anonymous;
}

static krb5_error_code
require_ext_opt(krb5_context context,
		krb5_get_init_creds_opt *opt,
		const char *type)
{
    if (opt->opt_private == NULL) {
	krb5_set_error_message(context, EINVAL,
			       N_("%s on non extendable opt", ""), type);
	return EINVAL;
    }
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_pa_password(krb5_context context,
					krb5_get_init_creds_opt *opt,
					const char *password,
					krb5_s2k_proc key_proc)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_pa_password");
    if (ret)
	return ret;
    opt->opt_private->password = password;
    opt->opt_private->key_proc = key_proc;
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_pac_request(krb5_context context,
					krb5_get_init_creds_opt *opt,
					krb5_boolean req_pac)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_pac_req");
    if (ret)
	return ret;
    opt->opt_private->req_pac = req_pac ?
	KRB5_INIT_CREDS_TRISTATE_TRUE :
	KRB5_INIT_CREDS_TRISTATE_FALSE;
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_addressless(krb5_context context,
					krb5_get_init_creds_opt *opt,
					krb5_boolean addressless)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_pac_req");
    if (ret)
	return ret;
    if (addressless)
	opt->opt_private->addressless = KRB5_INIT_CREDS_TRISTATE_TRUE;
    else
	opt->opt_private->addressless = KRB5_INIT_CREDS_TRISTATE_FALSE;
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_canonicalize(krb5_context context,
					 krb5_get_init_creds_opt *opt,
					 krb5_boolean req)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_canonicalize");
    if (ret)
	return ret;
    if (req)
	opt->opt_private->flags |= KRB5_INIT_CREDS_CANONICALIZE;
    else
	opt->opt_private->flags &= ~KRB5_INIT_CREDS_CANONICALIZE;
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_win2k(krb5_context context,
				  krb5_get_init_creds_opt *opt,
				  krb5_boolean req)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_win2k");
    if (ret)
	return ret;
    if (req) {
	opt->opt_private->flags |= KRB5_INIT_CREDS_NO_C_CANON_CHECK;
	opt->opt_private->flags |= KRB5_INIT_CREDS_NO_C_NO_EKU_CHECK;
	opt->opt_private->flags |= KRB5_INIT_CREDS_PKINIT_NO_KRBTGT_OTHERNAME_CHECK;
    } else {
	opt->opt_private->flags &= ~KRB5_INIT_CREDS_NO_C_CANON_CHECK;
	opt->opt_private->flags &= ~KRB5_INIT_CREDS_NO_C_NO_EKU_CHECK;
	opt->opt_private->flags &= ~KRB5_INIT_CREDS_PKINIT_NO_KRBTGT_OTHERNAME_CHECK;
    }
    return 0;
}


KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_set_process_last_req(krb5_context context,
					     krb5_get_init_creds_opt *opt,
					     krb5_gic_process_last_req func,
					     void *ctx)
{
    krb5_error_code ret;
    ret = require_ext_opt(context, opt, "init_creds_opt_set_win2k");
    if (ret)
	return ret;

    opt->opt_private->lr.func = func;
    opt->opt_private->lr.ctx = ctx;

    return 0;
}

#ifndef HEIMDAL_SMALLER

/**
 * Deprecated: use krb5_get_init_creds_opt_alloc().
 *
 * The reason krb5_get_init_creds_opt_init() is deprecated is that
 * krb5_get_init_creds_opt is a static structure and for ABI reason it
 * can't grow, ie can't add new functionality.
 *
 * @ingroup krb5_deprecated
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_get_init_creds_opt_init(krb5_get_init_creds_opt *opt)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    memset (opt, 0, sizeof(*opt));
}

/**
 * Deprecated: use the new krb5_init_creds_init() and
 * krb5_init_creds_get_error().
 *
 * @ingroup krb5_deprecated
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_init_creds_opt_get_error(krb5_context context,
				  krb5_get_init_creds_opt *opt,
				  KRB_ERROR **error)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    *error = calloc(1, sizeof(**error));
    if (*error == NULL) {
	krb5_set_error_message(context, ENOMEM, N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    return 0;
}

#endif /* HEIMDAL_SMALLER */
