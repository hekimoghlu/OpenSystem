/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
del_enctype(void *opt, int argc, char **argv)
{
    kadm5_principal_ent_rec princ;
    krb5_principal princ_ent = NULL;
    krb5_error_code ret;
    const char *princ_name;
    int i, j, k;
    krb5_key_data *new_key_data;
    int n_etypes;
    krb5_enctype *etypes;

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

    ret = kadm5_get_principal(kadm_handle, princ_ent, &princ,
			      KADM5_PRINCIPAL | KADM5_KEY_DATA);
    if (ret) {
	krb5_free_principal (context, princ_ent);
	krb5_warnx (context, "no such principal: %s", princ_name);
	goto out2;
    }

    new_key_data   = malloc(princ.n_key_data * sizeof(*new_key_data));
    if (new_key_data == NULL && princ.n_key_data != 0) {
	krb5_warnx (context, "out of memory");
	goto out;
    }

    for (i = 0, j = 0; i < princ.n_key_data; ++i) {
	krb5_key_data *key = &princ.key_data[i];
	int docopy = 1;

	for (k = 0; k < n_etypes; ++k)
	    if (etypes[k] == key->key_data_type[0]) {
		docopy = 0;
		break;
	    }
	if (docopy) {
	    new_key_data[j++] = *key;
	} else {
	    int16_t ignore = 1;

	    kadm5_free_key_data (kadm_handle, &ignore, key);
	}
    }

    free (princ.key_data);
    if (j == 0) {
	free(new_key_data);
	new_key_data = NULL;
    }
    princ.n_key_data = j;
    princ.key_data   = new_key_data;

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
