/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

/*
 * del_enctype principal enctypes...
 */

int
add_enctype(struct add_enctype_options*opt, int argc, char **argv)
{
    kadm5_principal_ent_rec princ;
    krb5_principal princ_ent = NULL;
    krb5_error_code ret;
    const char *princ_name;
    int i, j;
    krb5_key_data *new_key_data;
    int n_etypes;
    krb5_enctype *etypes;

    if (!opt->random_key_flag) {
	krb5_warnx (context, "only random key is supported now");
	return 0;
    }

    memset (&princ, 0, sizeof(princ));
    princ_name = argv[0];
    n_etypes   = argc - 1;
    etypes     = malloc (n_etypes * sizeof(*etypes));
    if (etypes == NULL) {
	krb5_warnx (context, "out of memory");
	return 0;
    }
    argv++;
    for (i = 0; i < n_etypes; ++i) {
	ret = krb5_string_to_enctype (context, argv[i], &etypes[i]);
	if (ret) {
	    krb5_warnx (context, "bad enctype \"%s\"", argv[i]);
	    goto out2;
	}
    }

    ret = krb5_parse_name(context, princ_name, &princ_ent);
    if (ret) {
	krb5_warn (context, ret, "krb5_parse_name %s", princ_name);
	goto out2;
    }

    /* The principal might have zero keys, but it will still have a kvno! */
    ret = kadm5_get_principal(kadm_handle, princ_ent, &princ,
			      KADM5_KVNO | KADM5_PRINCIPAL | KADM5_KEY_DATA);
    if (ret) {
	krb5_free_principal (context, princ_ent);
	krb5_warnx (context, "no such principal: %s", princ_name);
	goto out2;
    }

    new_key_data   = malloc((princ.n_key_data + n_etypes)
			    * sizeof(*new_key_data));
    if (new_key_data == NULL) {
	krb5_warnx (context, "out of memory");
	goto out;
    }

    for (i = 0; i < princ.n_key_data; ++i) {
	krb5_key_data *key = &princ.key_data[i];

	for (j = 0; j < n_etypes; ++j) {
	    if (etypes[j] == key->key_data_type[0]) {
		/* XXX Should this be an error?  The admin can del_enctype... */
		krb5_warnx(context, "enctype %d already exists",
			   (int)etypes[j]);
		free(new_key_data);
		goto out;
	    }
	}
	new_key_data[i] = *key;
    }

    for (i = 0; i < n_etypes; ++i) {
	int n = princ.n_key_data + i;
	krb5_keyblock keyblock;

	memset(&new_key_data[n], 0, sizeof(new_key_data[n]));
	new_key_data[n].key_data_ver = 2;
	new_key_data[n].key_data_kvno = princ.kvno;

	ret = krb5_generate_random_keyblock (context, etypes[i], &keyblock);
	if (ret) {
	    krb5_warnx(context, "genernate enctype %d failed", (int)etypes[i]);
	    while (--i >= 0)
		free(new_key_data[--n].key_data_contents[0]);
	    goto out;
	}

	/* key */
	new_key_data[n].key_data_type[0] = etypes[i];
	new_key_data[n].key_data_contents[0] = malloc(keyblock.keyvalue.length);
	if (new_key_data[n].key_data_contents[0] == NULL) {
	    ret = ENOMEM;
	    krb5_warn(context, ret, "out of memory");
	    while (--i >= 0)
		free(new_key_data[--n].key_data_contents[0]);
	    goto out;
	}
	new_key_data[n].key_data_length[0]   = keyblock.keyvalue.length;
	memcpy(new_key_data[n].key_data_contents[0],
	       keyblock.keyvalue.data,
	       keyblock.keyvalue.length);
	krb5_free_keyblock_contents(context, &keyblock);

	/* salt */
	new_key_data[n].key_data_type[1]     = KRB5_PW_SALT;
	new_key_data[n].key_data_length[1]   = 0;
	new_key_data[n].key_data_contents[1] = NULL;

    }

    free (princ.key_data);
    princ.n_key_data += n_etypes;
    princ.key_data   = new_key_data;
    new_key_data = NULL;

    ret = kadm5_modify_principal (kadm_handle, &princ, KADM5_KEY_DATA);
    if (ret)
	krb5_warn(context, ret, "kadm5_modify_principal");
out:
    krb5_free_principal (context, princ_ent);
    kadm5_free_principal_ent(kadm_handle, &princ);
out2:
    free (etypes);
    return ret != 0;
}
