/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#include <kadm5/private.h>

static kadm5_ret_t
create_random_entry(krb5_principal princ,
		    krb5_deltat max_life,
		    krb5_deltat max_rlife,
		    uint32_t attributes)
{
    kadm5_principal_ent_rec ent;
    kadm5_ret_t ret;
    int mask = 0;
    krb5_keyblock *keys;
    int n_keys, i;
    char *name;
    const char *password;
    char pwbuf[512];

    random_password(pwbuf, sizeof(pwbuf));
    password = pwbuf;

    ret = krb5_unparse_name(context, princ, &name);
    if (ret) {
	krb5_warn(context, ret, "failed to unparse principal name");
	return ret;
    }

    memset(&ent, 0, sizeof(ent));
    ent.principal = princ;
    mask |= KADM5_PRINCIPAL;
    if (max_life) {
	ent.max_life = max_life;
	mask |= KADM5_MAX_LIFE;
    }
    if (max_rlife) {
	ent.max_renewable_life = max_rlife;
	mask |= KADM5_MAX_RLIFE;
    }
    ent.attributes |= attributes | KRB5_KDB_DISALLOW_ALL_TIX;
    mask |= KADM5_ATTRIBUTES;

    /* Create the entry with a random password */
    ret = kadm5_create_principal(kadm_handle, &ent, mask, password);
    if(ret) {
	krb5_warn(context, ret, "create_random_entry(%s): randkey failed",
		  name);
	goto out;
    }

    /* Replace the string2key based keys with real random bytes */
    ret = kadm5_randkey_principal(kadm_handle, princ, &keys, &n_keys);
    if(ret) {
	krb5_warn(context, ret, "create_random_entry(%s): randkey failed",
		  name);
	goto out;
    }
    for(i = 0; i < n_keys; i++)
	krb5_free_keyblock_contents(context, &keys[i]);
    free(keys);
    ret = kadm5_get_principal(kadm_handle, princ, &ent,
			      KADM5_PRINCIPAL | KADM5_ATTRIBUTES);
    if(ret) {
	krb5_warn(context, ret, "create_random_entry(%s): "
		  "unable to get principal", name);
	goto out;
    }
    ent.attributes &= (~KRB5_KDB_DISALLOW_ALL_TIX);
    ent.kvno = 1;
    ret = kadm5_modify_principal(kadm_handle, &ent,
				 KADM5_ATTRIBUTES|KADM5_KVNO);
    kadm5_free_principal_ent (kadm_handle, &ent);
    if(ret) {
	krb5_warn(context, ret, "create_random_entry(%s): "
		  "unable to modify principal", name);
	goto out;
    }
 out:
    free(name);
    return ret;
}

extern int local_flag;

int
init(struct init_options *opt, int argc, char **argv)
{
    kadm5_ret_t ret;
    int i;
    HDB *db;
    krb5_deltat max_life = 0, max_rlife = 0;

    if (!local_flag) {
	krb5_warnx(context, "init is only available in local (-l) mode");
	return 0;
    }

    if (opt->realm_max_ticket_life_string) {
	if (str2deltat (opt->realm_max_ticket_life_string, &max_life) != 0) {
	    krb5_warnx (context, "unable to parse \"%s\"",
			opt->realm_max_ticket_life_string);
	    return 0;
	}
    }
    if (opt->realm_max_renewable_life_string) {
	if (str2deltat (opt->realm_max_renewable_life_string, &max_rlife) != 0) {
	    krb5_warnx (context, "unable to parse \"%s\"",
			opt->realm_max_renewable_life_string);
	    return 0;
	}
    }

    db = _kadm5_s_get_db(kadm_handle);

    ret = db->hdb_open(context, db, O_RDWR | O_CREAT, 0600);
    if(ret){
	krb5_warn(context, ret, "hdb_open");
	return 0;
    }
    db->hdb_close(context, db);
    for(i = 0; i < argc; i++){
	krb5_principal princ;
	const char *realm = argv[i];

	if (opt->realm_max_ticket_life_string == NULL) {
	    max_life = 0;
	    if(edit_deltat ("Realm max ticket life", &max_life, NULL, 0)) {
		return 0;
	    }
	}
	if (opt->realm_max_renewable_life_string == NULL) {
	    max_rlife = 0;
	    if(edit_deltat("Realm max renewable ticket life", &max_rlife,
			   NULL, 0)) {
		return 0;
	    }
	}

	/* Create `krbtgt/REALM' */
	ret = krb5_make_principal(context, &princ, realm,
				  KRB5_TGS_NAME, realm, NULL);
	if(ret)
	    return 0;

	create_random_entry(princ, max_life, max_rlife, 0);
	krb5_free_principal(context, princ);

	if (opt->bare_flag)
	    continue;

	/* Create `kadmin/changepw' */
	krb5_make_principal(context, &princ, realm,
			    "kadmin", "changepw", NULL);
	/*
	 * The Windows XP (at least) password changing protocol
	 * request the `kadmin/changepw' ticket with `renewable_ok,
	 * renewable, forwardable' and so fails if we disallow
	 * forwardable here.
	 */
	create_random_entry(princ, 5*60, 5*60,
			    KRB5_KDB_DISALLOW_TGT_BASED|
			    KRB5_KDB_PWCHANGE_SERVICE|
			    KRB5_KDB_DISALLOW_POSTDATED|
			    KRB5_KDB_DISALLOW_RENEWABLE|
			    KRB5_KDB_DISALLOW_PROXIABLE|
			    KRB5_KDB_REQUIRES_PRE_AUTH);
	krb5_free_principal(context, princ);

	/* Create `kadmin/admin' */
	krb5_make_principal(context, &princ, realm,
			    "kadmin", "admin", NULL);
	create_random_entry(princ, 60*60, 60*60, KRB5_KDB_REQUIRES_PRE_AUTH);
	krb5_free_principal(context, princ);

	/* Create `changepw/kerberos' (for v4 compat) */
	krb5_make_principal(context, &princ, realm,
			    "changepw", "kerberos", NULL);
	create_random_entry(princ, 60*60, 60*60,
			    KRB5_KDB_DISALLOW_TGT_BASED|
			    KRB5_KDB_PWCHANGE_SERVICE);

	krb5_free_principal(context, princ);

	/* Create `kadmin/hprop' for database propagation */
	krb5_make_principal(context, &princ, realm,
			    "kadmin", "hprop", NULL);
	create_random_entry(princ, 60*60, 60*60,
			    KRB5_KDB_REQUIRES_PRE_AUTH|
			    KRB5_KDB_DISALLOW_TGT_BASED);
	krb5_free_principal(context, princ);

	/* Create `WELLKNOWN/ANONYMOUS' for anonymous as-req */
	krb5_make_principal(context, &princ, realm,
			    KRB5_WELLKNOWN_NAME, KRB5_ANON_NAME, NULL);
	create_random_entry(princ, 60*60, 60*60,
			    KRB5_KDB_REQUIRES_PRE_AUTH);
	krb5_free_principal(context, princ);


	/* Create `WELLKNOWN/org.h5l.fast-cookie' for FAST cookie */
	krb5_make_principal(context, &princ, realm,
			    KRB5_WELLKNOWN_NAME, "org.h5l.fast-cookie", NULL);
	create_random_entry(princ, 60*60, 60*60,
			    KRB5_KDB_REQUIRES_PRE_AUTH|
			    KRB5_KDB_DISALLOW_TGT_BASED|
			    KRB5_KDB_DISALLOW_ALL_TIX);
	krb5_free_principal(context, princ);

	/* Create `default' */
	{
	    kadm5_principal_ent_rec ent;
	    int mask = 0;

	    memset (&ent, 0, sizeof(ent));
	    mask |= KADM5_PRINCIPAL;
	    krb5_make_principal(context, &ent.principal, realm,
				"default", NULL);
	    mask |= KADM5_MAX_LIFE;
	    ent.max_life = 24 * 60 * 60;
	    mask |= KADM5_MAX_RLIFE;
	    ent.max_renewable_life = 7 * ent.max_life;
	    ent.attributes = KRB5_KDB_DISALLOW_ALL_TIX;
	    mask |= KADM5_ATTRIBUTES;

	    ret = kadm5_create_principal(kadm_handle, &ent, mask, "");
	    if (ret)
		krb5_err (context, 1, ret, "kadm5_create_principal");

	    krb5_free_principal(context, ent.principal);
	}
    }
    return 0;
}

