/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#include "kadmin_locl.h"
#include "kadmin-commands.h"

static void
add_tl(kadm5_principal_ent_rec *princ, int type, krb5_data *data)
{
    krb5_tl_data *tl, **ptl;

    tl = ecalloc(1, sizeof(*tl));
    tl->tl_data_next = NULL;
    tl->tl_data_type = type;
    tl->tl_data_length = data->length;
    tl->tl_data_contents = data->data;

    princ->n_tl_data++;
    ptl = &princ->tl_data;
    while (*ptl != NULL)
	ptl = &(*ptl)->tl_data_next;
    *ptl = tl;

    return;
}

static void
add_constrained_delegation(krb5_context contextp,
			   kadm5_principal_ent_rec *princ,
			   struct getarg_strings *strings)
{
    krb5_error_code ret;
    HDB_extension ext;
    krb5_data buf;
    size_t size = 0;

    memset(&ext, 0, sizeof(ext));
    ext.mandatory = FALSE;
    ext.data.element = choice_HDB_extension_data_allowed_to_delegate_to;

    if (strings->num_strings == 1 && strings->strings[0][0] == '\0') {
	ext.data.u.allowed_to_delegate_to.val = NULL;
	ext.data.u.allowed_to_delegate_to.len = 0;
    } else {
	krb5_principal p;
	int i;

	ext.data.u.allowed_to_delegate_to.val =
	    calloc(strings->num_strings,
		   sizeof(ext.data.u.allowed_to_delegate_to.val[0]));
	ext.data.u.allowed_to_delegate_to.len = strings->num_strings;

	for (i = 0; i < strings->num_strings; i++) {
	    ret = krb5_parse_name(contextp, strings->strings[i], &p);
	    if (ret)
		abort();
	    ret = copy_Principal(p, &ext.data.u.allowed_to_delegate_to.val[i]);
	    if (ret)
		abort();
	    krb5_free_principal(contextp, p);
	}
    }

    ASN1_MALLOC_ENCODE(HDB_extension, buf.data, buf.length,
		       &ext, &size, ret);
    free_HDB_extension(&ext);
    if (ret)
	abort();
    if (buf.length != size)
	abort();

    add_tl(princ, KRB5_TL_EXTENSION, &buf);
}

static void
add_aliases(krb5_context contextp, kadm5_principal_ent_rec *princ,
	    struct getarg_strings *strings)
{
    krb5_error_code ret;
    HDB_extension ext;
    krb5_data buf;
    krb5_principal p;
    size_t size = 0;
    int i;

    memset(&ext, 0, sizeof(ext));
    ext.mandatory = FALSE;
    ext.data.element = choice_HDB_extension_data_aliases;
    ext.data.u.aliases.case_insensitive = 0;

    if (strings->num_strings == 1 && strings->strings[0][0] == '\0') {
	ext.data.u.aliases.aliases.val = NULL;
	ext.data.u.aliases.aliases.len = 0;
    } else {
	ext.data.u.aliases.aliases.val =
	    calloc(strings->num_strings,
		   sizeof(ext.data.u.aliases.aliases.val[0]));
	ext.data.u.aliases.aliases.len = strings->num_strings;

	for (i = 0; i < strings->num_strings; i++) {
	    (void)krb5_parse_name(contextp, strings->strings[i], &p);
	    (void)copy_Principal(p, &ext.data.u.aliases.aliases.val[i]);
	    krb5_free_principal(contextp, p);
	}
    }

    ASN1_MALLOC_ENCODE(HDB_extension, buf.data, buf.length,
		       &ext, &size, ret);
    free_HDB_extension(&ext);
    if (ret)
	abort();
    if (buf.length != size)
	abort();

    add_tl(princ, KRB5_TL_EXTENSION, &buf);
}

static void
add_pkinit_acl(krb5_context contextp, kadm5_principal_ent_rec *princ,
	       struct getarg_strings *strings)
{
    krb5_error_code ret;
    HDB_extension ext;
    krb5_data buf;
    size_t size = 0;
    int i;

    memset(&ext, 0, sizeof(ext));
    ext.mandatory = FALSE;
    ext.data.element = choice_HDB_extension_data_pkinit_acl;
    ext.data.u.aliases.case_insensitive = 0;

    if (strings->num_strings == 1 && strings->strings[0][0] == '\0') {
	ext.data.u.pkinit_acl.val = NULL;
	ext.data.u.pkinit_acl.len = 0;
    } else {
	ext.data.u.pkinit_acl.val =
	    calloc(strings->num_strings,
		   sizeof(ext.data.u.pkinit_acl.val[0]));
	ext.data.u.pkinit_acl.len = strings->num_strings;

	for (i = 0; i < strings->num_strings; i++) {
	    ext.data.u.pkinit_acl.val[i].subject = estrdup(strings->strings[i]);
	}
    }

    ASN1_MALLOC_ENCODE(HDB_extension, buf.data, buf.length,
		       &ext, &size, ret);
    free_HDB_extension(&ext);
    if (ret)
	abort();
    if (buf.length != size)
	abort();

    add_tl(princ, KRB5_TL_EXTENSION, &buf);
}

static void
add_kvno_diff(krb5_context contextp, kadm5_principal_ent_rec *princ,
	      int is_svc_diff, krb5_kvno kvno_diff)
{
    krb5_error_code ret;
    HDB_extension ext;
    krb5_data buf;
    size_t size = 0;

    if (kvno_diff < 0)
	return;
    if (kvno_diff > 2048)
	kvno_diff = 2048;

    if (is_svc_diff) {
	ext.data.element = choice_HDB_extension_data_hist_kvno_diff_svc;
	ext.data.u.hist_kvno_diff_svc = (unsigned int)kvno_diff;
    } else {
	ext.data.element = choice_HDB_extension_data_hist_kvno_diff_clnt;
	ext.data.u.hist_kvno_diff_clnt = (unsigned int)kvno_diff;
    }
    ASN1_MALLOC_ENCODE(HDB_extension, buf.data, buf.length,
		       &ext, &size, ret);
    if (ret)
	abort();
    if (buf.length != size)
	abort();

    add_tl(princ, KRB5_TL_EXTENSION, &buf);
}

static int
do_mod_entry(krb5_principal principal, void *data)
{
    krb5_error_code ret;
    kadm5_principal_ent_rec princ;
    int mask = 0;
    struct modify_options *e = data;

    memset (&princ, 0, sizeof(princ));
    ret = kadm5_get_principal(kadm_handle, principal, &princ,
			      KADM5_PRINCIPAL | KADM5_ATTRIBUTES |
			      KADM5_MAX_LIFE | KADM5_MAX_RLIFE |
			      KADM5_PRINC_EXPIRE_TIME |
			      KADM5_PW_EXPIRATION);
    if(ret)
	return ret;

    if(e->max_ticket_life_string ||
       e->max_renewable_life_string ||
       e->expiration_time_string ||
       e->pw_expiration_time_string ||
       e->attributes_string ||
       e->policy_string ||
       e->kvno_integer != -1 ||
       e->constrained_delegation_strings.num_strings ||
       e->alias_strings.num_strings ||
       e->pkinit_acl_strings.num_strings ||
       e->hist_kvno_diff_clnt_integer != -1 ||
       e->hist_kvno_diff_svc_integer != -1) {
	ret = set_entry(context, &princ, &mask,
			e->max_ticket_life_string,
			e->max_renewable_life_string,
			e->expiration_time_string,
			e->pw_expiration_time_string,
			e->attributes_string,
			e->policy_string);
	if(e->kvno_integer != -1) {
	    princ.kvno = e->kvno_integer;
	    mask |= KADM5_KVNO;
	}
	if (e->constrained_delegation_strings.num_strings) {
	    add_constrained_delegation(context, &princ,
				       &e->constrained_delegation_strings);
	    mask |= KADM5_TL_DATA;
	}
	if (e->alias_strings.num_strings) {
	    add_aliases(context, &princ, &e->alias_strings);
	    mask |= KADM5_TL_DATA;
	}
	if (e->pkinit_acl_strings.num_strings) {
	    add_pkinit_acl(context, &princ, &e->pkinit_acl_strings);
	    mask |= KADM5_TL_DATA;
	}
	if (e->hist_kvno_diff_clnt_integer != -1) {
	    add_kvno_diff(context, &princ, 0, e->hist_kvno_diff_clnt_integer);
	    mask |= KADM5_TL_DATA;
	}
	if (e->hist_kvno_diff_svc_integer != -1) {
	    add_kvno_diff(context, &princ, 1, e->hist_kvno_diff_clnt_integer);
	    mask |= KADM5_TL_DATA;
	}
    } else
	ret = edit_entry(&princ, &mask, NULL, 0);
    if(ret == 0) {
	ret = kadm5_modify_principal(kadm_handle, &princ, mask);
	if(ret)
	    krb5_warn(context, ret, "kadm5_modify_principal");
    }

    kadm5_free_principal_ent(kadm_handle, &princ);
    return ret;
}

int
mod_entry(struct modify_options *opt, int argc, char **argv)
{
    krb5_error_code ret = 0;
    int i;

    for(i = 0; i < argc; i++) {
	ret = foreach_principal(argv[i], do_mod_entry, "mod", opt);
	if (ret)
	    break;
    }
    return ret != 0;
}

