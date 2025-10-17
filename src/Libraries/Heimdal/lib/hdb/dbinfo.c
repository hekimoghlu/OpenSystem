/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "hdb_locl.h"

struct hdb_dbinfo {
    char *label;
    char *realm;
    char *dbname;
    char *mkey_file;
    char *acl_file;
    char *log_file;
    const krb5_config_binding *binding;
    struct hdb_dbinfo *next;
};

static int
get_dbinfo(krb5_context context,
	   const krb5_config_binding *db_binding,
	   const char *label,
	   struct hdb_dbinfo **db)
{
    struct hdb_dbinfo *di;
    const char *p;

    *db = NULL;

    p = krb5_config_get_string(context, db_binding, "dbname", NULL);
    if(p == NULL)
	return 0;

    di = calloc(1, sizeof(*di));
    if (di == NULL) {
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }
    di->label = strdup(label);
    di->dbname = strdup(p);

    p = krb5_config_get_string(context, db_binding, "realm", NULL);
    if(p)
	di->realm = strdup(p);
    p = krb5_config_get_string(context, db_binding, "mkey_file", NULL);
    if(p)
	di->mkey_file = strdup(p);
    p = krb5_config_get_string(context, db_binding, "acl_file", NULL);
    if(p)
	di->acl_file = strdup(p);
    p = krb5_config_get_string(context, db_binding, "log_file", NULL);
    if(p)
	di->log_file = strdup(p);

    di->binding = db_binding;

    *db = di;
    return 0;
}


int
hdb_get_dbinfo(krb5_context context, struct hdb_dbinfo **dbp)
{
    const krb5_config_binding *db_binding;
    struct hdb_dbinfo *di, **dt, *databases;
    const char *default_dbname = HDB_DEFAULT_DB;
    const char *default_mkey = HDB_DB_DIR "/m-key";
    const char *default_acl = HDB_DB_DIR "/kadmind.acl";
    const char *p;
    int ret;

    *dbp = NULL;
    dt = NULL;
    databases = NULL;

    db_binding = krb5_config_get_list(context, NULL,
				      "kdc",
				      "database",
				      NULL);
    if (db_binding) {

	ret = get_dbinfo(context, db_binding, "default", &di);
	if (ret == 0 && di) {
	    databases = di;
	    dt = &di->next;
	}

	for ( ; db_binding != NULL; db_binding = db_binding->next) {

	    if (db_binding->type != krb5_config_list)
		continue;

	    ret = get_dbinfo(context, db_binding->u.list,
			     db_binding->name, &di);
	    if (ret)
		krb5_err(context, 1, ret, "failed getting realm");

	    if (di == NULL)
		continue;

	    if (dt)
		*dt = di;
	    else
		databases = di;
	    dt = &di->next;

	}
    }

    if(databases == NULL) {
	/* if there are none specified, create one and use defaults */
	di = calloc(1, sizeof(*di));
	databases = di;
	di->label = strdup("default");
    }

    for(di = databases; di; di = di->next) {
	if(di->dbname == NULL) {
	    di->dbname = strdup(default_dbname);
	    if (di->mkey_file == NULL)
		di->mkey_file = strdup(default_mkey);
	}
	if(di->mkey_file == NULL) {
	    p = strrchr(di->dbname, '.');
	    if(p == NULL || strchr(p, '/') != NULL)
		/* final pathname component does not contain a . */
		asprintf(&di->mkey_file, "%s.mkey", di->dbname);
	    else
		/* the filename is something.else, replace .else with
                   .mkey */
		asprintf(&di->mkey_file, "%.*s.mkey",
			 (int)(p - di->dbname), di->dbname);
	}
	if(di->acl_file == NULL)
	    di->acl_file = strdup(default_acl);
    }
    *dbp = databases;
    return 0;
}


struct hdb_dbinfo *
hdb_dbinfo_get_next(struct hdb_dbinfo *dbp, struct hdb_dbinfo *dbprevp)
{
    if (dbprevp == NULL)
	return dbp;
    else
	return dbprevp->next;
}

const char *
hdb_dbinfo_get_label(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->label;
}

const char *
hdb_dbinfo_get_realm(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->realm;
}

const char *
hdb_dbinfo_get_dbname(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->dbname;
}

const char *
hdb_dbinfo_get_mkey_file(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->mkey_file;
}

const char *
hdb_dbinfo_get_acl_file(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->acl_file;
}

const char *
hdb_dbinfo_get_log_file(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->log_file;
}

const krb5_config_binding *
hdb_dbinfo_get_binding(krb5_context context, struct hdb_dbinfo *dbp)
{
    return dbp->binding;
}

void
hdb_free_dbinfo(krb5_context context, struct hdb_dbinfo **dbp)
{
    struct hdb_dbinfo *di, *ndi;

    for(di = *dbp; di != NULL; di = ndi) {
	ndi = di->next;
	free (di->label);
	free (di->realm);
	free (di->dbname);
	free (di->mkey_file);
	free (di->acl_file);
	free (di->log_file);
	free(di);
    }
    *dbp = NULL;
}

/**
 * Return the directory where the hdb database resides.
 *
 * @param context Kerberos 5 context.
 *
 * @return string pointing to directory.
 */

const char *
hdb_db_dir(krb5_context context)
{
    return HDB_DB_DIR;
}

/**
 * Return the default hdb database resides.
 *
 * @param context Kerberos 5 context.
 *
 * @return string pointing to directory.
 */

const char *
hdb_default_db(krb5_context context)
{
    return HDB_DEFAULT_DB;
}
