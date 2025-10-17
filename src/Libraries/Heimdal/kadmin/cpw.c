/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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

struct cpw_entry_data {
    int keepold;
    int random_key;
    int random_password;
    char *password;
    krb5_key_data *key_data;
};

static int
set_random_key (krb5_principal principal, int keepold)
{
    krb5_error_code ret;
    int i;
    krb5_keyblock *keys;
    int num_keys;

    ret = kadm5_randkey_principal_3(kadm_handle, principal, keepold, 0, NULL,
				    &keys, &num_keys);
    if(ret)
	return ret;
    for(i = 0; i < num_keys; i++)
	krb5_free_keyblock_contents(context, &keys[i]);
    free(keys);
    return 0;
}

static int
set_random_password (krb5_principal principal, int keepold)
{
    krb5_error_code ret;
    char pw[128];

    random_password (pw, sizeof(pw));
    ret = kadm5_chpass_principal_3(kadm_handle, principal, keepold, 0, NULL, pw);
    if (ret == 0) {
	char *princ_name;

	krb5_unparse_name(context, principal, &princ_name);

	printf ("%s's password set to \"%s\"\n", princ_name, pw);
	free (princ_name);
    }
    memset (pw, 0, sizeof(pw));
    return ret;
}

static int
set_password (krb5_principal principal, char *password, int keepold)
{
    krb5_error_code ret = 0;
    char pwbuf[128];

    if(password == NULL) {
	char *princ_name;
	char *prompt;

	krb5_unparse_name(context, principal, &princ_name);
	asprintf(&prompt, "%s's Password: ", princ_name);
	free (princ_name);
	ret = UI_UTIL_read_pw_string(pwbuf, sizeof(pwbuf), prompt, 1);
	free (prompt);
	if(ret){
	    return 0; /* XXX error code? */
	}
	password = pwbuf;
    }
    if(ret == 0)
	ret = kadm5_chpass_principal_3(kadm_handle, principal, keepold, 0, NULL,
				       password);
    memset(pwbuf, 0, sizeof(pwbuf));
    return ret;
}

static int
set_key_data (krb5_principal principal, krb5_key_data *key_data, int keepold)
{
    krb5_error_code ret;

    ret = kadm5_chpass_principal_with_key_3(kadm_handle, principal, keepold,
					    3, key_data);
    return ret;
}

static int
do_cpw_entry(krb5_principal principal, void *data)
{
    struct cpw_entry_data *e = data;

    if (e->random_key)
	return set_random_key (principal, e->keepold);
    else if (e->random_password)
	return set_random_password (principal, e->keepold);
    else if (e->key_data)
	return set_key_data (principal, e->key_data, e->keepold);
    else
	return set_password (principal, e->password, e->keepold);
}

int
cpw_entry(struct passwd_options *opt, int argc, char **argv)
{
    krb5_error_code ret = 0;
    int i;
    struct cpw_entry_data data;
    int num;
    krb5_key_data key_data[3];

    data.keepold = opt->keepold_flag;
    data.random_key = opt->random_key_flag;
    data.random_password = opt->random_password_flag;
    data.password = opt->password_string;
    data.key_data	 = NULL;

    num = 0;
    if (data.random_key)
	++num;
    if (data.random_password)
	++num;
    if (data.password)
	++num;
    if (opt->key_string)
	++num;

    if (num > 1) {
	fprintf (stderr, "give only one of "
		"--random-key, --random-password, --password, --key\n");
	return 1;
    }

    if (opt->key_string) {
	const char *error;

	if (parse_des_key (opt->key_string, key_data, &error)) {
	    fprintf (stderr, "failed parsing key \"%s\": %s\n",
		     opt->key_string, error);
	    return 1;
	}
	data.key_data = key_data;
    }

    for(i = 0; i < argc; i++)
	ret = foreach_principal(argv[i], do_cpw_entry, "cpw", &data);

    if (data.key_data) {
	int16_t dummy;
	kadm5_free_key_data (kadm_handle, &dummy, key_data);
    }

    return ret != 0;
}
