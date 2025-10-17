/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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

struct ext_keytab_data {
    krb5_keytab keytab;
};

static int
do_ext_keytab(krb5_principal principal, void *data)
{
    krb5_error_code ret;
    kadm5_principal_ent_rec princ;
    struct ext_keytab_data *e = data;
    krb5_keytab_entry *keys = NULL;
    krb5_keyblock *k = NULL;
    int i, n_k;
    int have_keys = 0;

    ret = kadm5_get_principal(kadm_handle, principal, &princ,
			      KADM5_PRINCIPAL|KADM5_KVNO|KADM5_KEY_DATA);
    if(ret)
	return ret;

    /* 
     * Make sure that we have keys, some backends doesn't support
     * getting keys, so in that case we fallback to
     * kadm5_randkey_principal() instead.
     */
    for (i = 0; i < princ.n_key_data; i++)
	if (princ.key_data[i].key_data_length[0] > 0)
	    have_keys = 1;

    if (princ.n_key_data && have_keys) {
	keys = malloc(sizeof(*keys) * princ.n_key_data);
	if (keys == NULL) {
	    kadm5_free_principal_ent(kadm_handle, &princ);
	    krb5_clear_error_message(context);
	    return ENOMEM;
	}
	for (i = 0; i < princ.n_key_data; i++) {
	    krb5_key_data *kd = &princ.key_data[i];

	    keys[i].principal = princ.principal;
	    keys[i].vno = kd->key_data_kvno;
	    keys[i].keyblock.keytype = kd->key_data_type[0];
	    keys[i].keyblock.keyvalue.length = kd->key_data_length[0];
	    keys[i].keyblock.keyvalue.data = kd->key_data_contents[0];
	    keys[i].timestamp = (uint32_t)time(NULL);
	}

	n_k = princ.n_key_data;
    } else {
	ret = kadm5_randkey_principal(kadm_handle, principal, &k, &n_k);
	if (ret) {
	    kadm5_free_principal_ent(kadm_handle, &princ);
	    return ret;
	}
	keys = malloc(sizeof(*keys) * n_k);
	if (keys == NULL) {
	    kadm5_free_principal_ent(kadm_handle, &princ);
	    krb5_clear_error_message(context);
	    return ENOMEM;
	}
	for (i = 0; i < n_k; i++) {
	    keys[i].principal = principal;
	    keys[i].vno = princ.kvno + 1; /* XXX get entry again */
	    keys[i].keyblock = k[i];
	    keys[i].timestamp = (uint32_t)time(NULL);
	}
    }

    for(i = 0; i < n_k; i++) {
	ret = krb5_kt_add_entry(context, e->keytab, &keys[i]);
	if(ret)
	    krb5_warn(context, ret, "krb5_kt_add_entry(%d)", i);
    }

    if (k) {
	memset(k, 0, n_k * sizeof(*k));
	free(k);
    }
    if (keys)
	free(keys);
    kadm5_free_principal_ent(kadm_handle, &princ);
    return 0;
}

int
ext_keytab(struct ext_keytab_options *opt, int argc, char **argv)
{
    krb5_error_code ret;
    int i;
    struct ext_keytab_data data;

    if (opt->keytab_string == NULL)
	ret = krb5_kt_default(context, &data.keytab);
    else
	ret = krb5_kt_resolve(context, opt->keytab_string, &data.keytab);

    if(ret){
	krb5_warn(context, ret, "krb5_kt_resolve");
	return 1;
    }

    for(i = 0; i < argc; i++) {
	ret = foreach_principal(argv[i], do_ext_keytab, "ext", &data);
	if (ret)
	    break;
    }

    krb5_kt_close(context, data.keytab);

    return ret != 0;
}
