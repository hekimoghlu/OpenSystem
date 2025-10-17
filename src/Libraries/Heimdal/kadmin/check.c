/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
/*
 * Check database for strange configurations on default principals
 */

#include "kadmin_locl.h"
#include "kadmin-commands.h"

static int
get_check_entry(const char *name, kadm5_principal_ent_rec *ent)
{
    krb5_error_code ret;
    krb5_principal principal;

    ret = krb5_parse_name(context, name, &principal);
    if (ret) {
	krb5_warn(context, ret, "krb5_unparse_name: %s", name);
	return 1;
    }

    memset(ent, 0, sizeof(*ent));
    ret = kadm5_get_principal(kadm_handle, principal, ent, 0);
    krb5_free_principal(context, principal);
    if(ret)
	return 1;

    return 0;
}


static int
do_check_entry(krb5_principal principal, void *data)
{
    krb5_error_code ret;
    kadm5_principal_ent_rec princ;
    char *name;
    int i;

    ret = krb5_unparse_name(context, principal, &name);
    if (ret)
	return 1;

    memset (&princ, 0, sizeof(princ));
    ret = kadm5_get_principal(kadm_handle, principal, &princ,
			      KADM5_PRINCIPAL | KADM5_KEY_DATA);
    if(ret) {
	krb5_warn(context, ret, "Failed to get principal: %s", name);
	free(name);
	return 0;
    }

    for (i = 0; i < princ.n_key_data; i++) {
	size_t keysize;
	ret = krb5_enctype_keysize(context,
				   princ.key_data[i].key_data_type[0],
				   &keysize);
	if (ret == 0 && keysize != (size_t)princ.key_data[i].key_data_length[0]) {
	    krb5_warnx(context,
		       "Principal %s enctype %d, wrong length: %lu\n",
		       name, princ.key_data[i].key_data_type[0],
		       (unsigned long)princ.key_data[i].key_data_length);
	}
    }

    free(name);
    kadm5_free_principal_ent(kadm_handle, &princ);

    return 0;
}

int
check(struct check_options *opt, int argc, char **argv)
{
    kadm5_principal_ent_rec ent;
    krb5_error_code ret;
    char *realm = NULL, *p, *p2;
    int found;

    if (argc == 0) {
	ret = krb5_get_default_realm(context, &realm);
	if (ret) {
	    krb5_warn(context, ret, "krb5_get_default_realm");
	    goto fail;
	}
    } else {
	realm = strdup(argv[0]);
	if (realm == NULL) {
	    krb5_warnx(context, "malloc");
	    goto fail;
	}
    }

    /*
     * Check krbtgt/REALM@REALM
     *
     * For now, just check existance
     */

    if (asprintf(&p, "%s/%s@%s", KRB5_TGS_NAME, realm, realm) == -1) {
	krb5_warn(context, errno, "asprintf");
	goto fail;
    }

    ret = get_check_entry(p, &ent);
    if (ret) {
	printf("%s doesn't exist, are you sure %s is a realm in your database",
	       p, realm);
	free(p);
	goto fail;
    }
    free(p);

    kadm5_free_principal_ent(kadm_handle, &ent);

    /*
     * Check kadmin/admin@REALM
     */

    if (!opt->ds_local_flag) {
	if (asprintf(&p, "kadmin/admin@%s", realm) == -1) {
	    krb5_warn(context, errno, "asprintf");
	    goto fail;
	}

	ret = get_check_entry(p, &ent);
	if (ret) {
	    printf("%s doesn't exist, "
		   "there is no way to do remote administration", p);
	    free(p);
	    goto fail;
	}
	free(p);

	kadm5_free_principal_ent(kadm_handle, &ent);
    }

    /*
     * Check kadmin/changepw@REALM
     */

    if (!opt->ds_local_flag) {
	if (asprintf(&p, "kadmin/changepw@%s", realm) == -1) {
	    krb5_warn(context, errno, "asprintf");
	    goto fail;
	}

	ret = get_check_entry(p, &ent);
	if (ret) {
	    printf("%s doesn't exist, "
		   "there is no way to do change password", p);
	    free(p);
	    goto fail;
	}
	free(p);

	kadm5_free_principal_ent(kadm_handle, &ent);
    }

    /*
     * Check for duplicate afs keys
     */

    p2 = strdup(realm);
    if (p2 == NULL) {
	krb5_warn(context, errno, "malloc");
	goto fail;
    }
    strlwr(p2);

    if (asprintf(&p, "afs/%s@%s", p2, realm) == -1) {
	krb5_warn(context, errno, "asprintf");
	free(p2);
	goto fail;
    }
    free(p2);

    ret = get_check_entry(p, &ent);
    free(p);
    if (ret == 0) {
	kadm5_free_principal_ent(kadm_handle, &ent);
	found = 1;
    } else
	found = 0;

    if (asprintf(&p, "afs@%s", realm) == -1) {
	krb5_warn(context, errno, "asprintf");
	goto fail;
    }

    ret = get_check_entry(p, &ent);
    free(p);
    if (ret == 0) {
	kadm5_free_principal_ent(kadm_handle, &ent);
	if (found) {
	    krb5_warnx(context, "afs@REALM and afs/cellname@REALM both exists");
	    goto fail;
	}
    }

    if (!opt->ds_local_flag) {
	foreach_principal("*", do_check_entry, "check", NULL);
    }

    free(realm);
    return 0;
fail:
    free(realm);
    return 1;
}
