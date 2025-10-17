/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "ktutil_locl.h"

RCSID("$Id$");

static krb5_error_code
change_entry (krb5_keytab keytab,
	      krb5_principal principal, krb5_kvno kvno,
	      const char *realm, const char *admin_server, int server_port)
{
    krb5_error_code ret;
    kadm5_config_params conf;
    void *kadm_handle;
    char *client_name;
    krb5_keyblock *keys;
    int num_keys;
    int i;

    ret = krb5_unparse_name (context, principal, &client_name);
    if (ret) {
	krb5_warn (context, ret, "krb5_unparse_name");
	return ret;
    }

    memset (&conf, 0, sizeof(conf));

    if(realm == NULL)
	realm = krb5_principal_get_realm(context, principal);
    conf.realm = strdup(realm);
    if (conf.realm == NULL) {
	free (client_name);
	krb5_set_error_message(context, ENOMEM, "malloc failed");
	return ENOMEM;
    }
    conf.mask |= KADM5_CONFIG_REALM;

    if (admin_server) {
	conf.admin_server = strdup(admin_server);
	if (conf.admin_server == NULL) {
	    free(client_name);
	    free(conf.realm);
	    krb5_set_error_message(context, ENOMEM, "malloc failed");
	    return ENOMEM;
	}
	conf.mask |= KADM5_CONFIG_ADMIN_SERVER;
    }

    if (server_port) {
	conf.kadmind_port = htons(server_port);
	conf.mask |= KADM5_CONFIG_KADMIND_PORT;
    }

    ret = kadm5_init_with_skey_ctx (context,
				    client_name,
				    keytab_string,
				    KADM5_ADMIN_SERVICE,
				    &conf, 0, 0,
				    &kadm_handle);
    free(conf.admin_server);
    free(conf.realm);
    if (ret) {
	krb5_warn (context, ret,
		   "kadm5_c_init_with_skey_ctx: %s:", client_name);
	free (client_name);
	return ret;
    }
    ret = kadm5_randkey_principal (kadm_handle, principal, &keys, &num_keys);
    kadm5_destroy (kadm_handle);
    if (ret) {
	krb5_warn(context, ret, "kadm5_randkey_principal: %s:", client_name);
	free (client_name);
	return ret;
    }
    free (client_name);
    for (i = 0; i < num_keys; ++i) {
	krb5_keytab_entry new_entry;

	new_entry.principal = principal;
	new_entry.timestamp = (uint32_t)time (NULL);
	new_entry.vno = kvno + 1;
	new_entry.keyblock  = keys[i];

	ret = krb5_kt_add_entry (context, keytab, &new_entry);
	if (ret)
	    krb5_warn (context, ret, "krb5_kt_add_entry");
	krb5_free_keyblock_contents (context, &keys[i]);
    }
    return ret;
}

/*
 * loop over all the entries in the keytab (or those given) and change
 * their keys, writing the new keys
 */

struct change_set {
    krb5_principal principal;
    krb5_kvno kvno;
};

int
kt_change (struct change_options *opt, int argc, char **argv)
{
    krb5_error_code ret;
    krb5_keytab keytab;
    krb5_kt_cursor cursor;
    krb5_keytab_entry entry;
    int i, j, max;
    struct change_set *changeset;
    int errors = 0;

    if((keytab = ktutil_open_keytab()) == NULL)
	return 1;

    j = 0;
    max = 0;
    changeset = NULL;

    ret = krb5_kt_start_seq_get(context, keytab, &cursor);
    if(ret){
	krb5_warn(context, ret, "%s", keytab_string);
	goto out;
    }

    while((ret = krb5_kt_next_entry(context, keytab, &entry, &cursor)) == 0) {
	int add = 0;

	for (i = 0; i < j; ++i) {
	    if (krb5_principal_compare (context, changeset[i].principal,
					entry.principal)) {
		if (changeset[i].kvno < entry.vno)
		    changeset[i].kvno = entry.vno;
		break;
	    }
	}
	if (i < j) {
	    krb5_kt_free_entry (context, &entry);
	    continue;
	}

	if (argc == 0) {
	    add = 1;
	} else {
	    for (i = 0; i < argc; ++i) {
		krb5_principal princ;

		ret = krb5_parse_name (context, argv[i], &princ);
		if (ret) {
		    krb5_warn (context, ret, "%s", argv[i]);
		    continue;
		}
		if (krb5_principal_compare (context, princ, entry.principal))
		    add = 1;

		krb5_free_principal (context, princ);
	    }
	}

	if (add) {
	    if (j >= max) {
		void *tmp;

		max = max(max * 2, 1);
		tmp = realloc (changeset, max * sizeof(*changeset));
		if (tmp == NULL) {
		    krb5_kt_free_entry (context, &entry);
		    krb5_warnx (context, "realloc: out of memory");
		    ret = ENOMEM;
		    break;
		}
		changeset = tmp;
	    }
	    ret = krb5_copy_principal (context, entry.principal,
				       &changeset[j].principal);
	    if (ret) {
		krb5_warn (context, ret, "krb5_copy_principal");
		krb5_kt_free_entry (context, &entry);
		break;
	    }
	    changeset[j].kvno = entry.vno;
	    ++j;
	}
	krb5_kt_free_entry (context, &entry);
    }
    krb5_kt_end_seq_get(context, keytab, &cursor);

    if (ret == KRB5_KT_END) {
	for (i = 0; i < j; i++) {
	    if (verbose_flag) {
		char *client_name;

		ret = krb5_unparse_name (context, changeset[i].principal,
					 &client_name);
		if (ret) {
		    krb5_warn (context, ret, "krb5_unparse_name");
		} else {
		    printf("Changing %s kvno %d\n",
			   client_name, changeset[i].kvno);
		    free(client_name);
		}
	    }
	    ret = change_entry (keytab,
				changeset[i].principal, changeset[i].kvno,
				opt->realm_string,
				opt->admin_server_string,
				opt->server_port_integer);
	    if (ret != 0)
		errors = 1;
	}
    } else
	errors = 1;
    for (i = 0; i < j; i++)
	krb5_free_principal (context, changeset[i].principal);
    free (changeset);

 out:
    krb5_kt_close(context, keytab);
    return errors;
}
