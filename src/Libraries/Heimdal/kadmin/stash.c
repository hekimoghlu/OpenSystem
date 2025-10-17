/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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

extern int local_flag;

int
stash(struct stash_options *opt, int argc, char **argv)
{
    char buf[1024];
    krb5_error_code ret;
    krb5_enctype enctype;
    hdb_master_key mkey;

    if(!local_flag) {
	krb5_warnx(context, "stash is only available in local (-l) mode");
	return 0;
    }

    ret = krb5_string_to_enctype(context, opt->enctype_string, &enctype);
    if(ret) {
	krb5_warn(context, ret, "%s", opt->enctype_string);
	return 0;
    }

    if(opt->key_file_string == NULL) {
	asprintf(&opt->key_file_string, "%s/m-key", hdb_db_dir(context));
	if (opt->key_file_string == NULL)
	    errx(1, "out of memory");
    }

    ret = hdb_read_master_key(context, opt->key_file_string, &mkey);
    if(ret && ret != ENOENT) {
	krb5_warn(context, ret, "reading master key from %s",
		  opt->key_file_string);
	return 0;
    }

    if (opt->convert_file_flag) {
	if (ret)
	    krb5_warn(context, ret, "reading master key from %s",
		      opt->key_file_string);
	return 0;
    } else {
	krb5_keyblock key;
	krb5_salt salt;
	salt.salttype = KRB5_PW_SALT;
	/* XXX better value? */
	salt.saltvalue.data = NULL;
	salt.saltvalue.length = 0;
	if(opt->master_key_fd_integer != -1) {
	    ssize_t n;
	    n = read(opt->master_key_fd_integer, buf, sizeof(buf));
	    if(n == 0)
		krb5_warnx(context, "end of file reading passphrase");
	    else if(n < 0) {
		krb5_warn(context, errno, "reading passphrase");
		n = 0;
	    }
	    buf[n] = '\0';
	    buf[strcspn(buf, "\r\n")] = '\0';
	} else if (opt->random_password_flag) {
	    random_password (buf, sizeof(buf));
	    if (opt->print_password_flag)
		printf("Using random master stash password: %s\n", buf);
	} else {
	    if(UI_UTIL_read_pw_string(buf, sizeof(buf), "Master key: ", 1)) {
		hdb_free_master_key(context, mkey);
		return 0;
	    }
	}
	(void)krb5_string_to_key_salt(context, enctype, buf, salt, &key);
	(void)hdb_add_master_key(context, &key, &mkey);
	krb5_free_keyblock_contents(context, &key);
    }

    {
	char *new, *old;
	asprintf(&old, "%s.old", opt->key_file_string);
	asprintf(&new, "%s.new", opt->key_file_string);
	if(old == NULL || new == NULL) {
	    ret = ENOMEM;
	    goto out;
	}

	if(unlink(new) < 0 && errno != ENOENT) {
	    ret = errno;
	    goto out;
	}
	krb5_warnx(context, "writing key to \"%s\"", opt->key_file_string);
	ret = hdb_write_master_key(context, new, mkey);
	if(ret)
	    unlink(new);
	else {
	    unlink(old);
#ifndef NO_POSIX_LINKS
	    if(link(opt->key_file_string, old) < 0 && errno != ENOENT) {
		ret = errno;
		unlink(new);
	    } else {
#endif
		if(rename(new, opt->key_file_string) < 0) {
		    ret = errno;
		}
#ifndef NO_POSIX_LINKS
	    }
#endif
	}
    out:
	free(old);
	free(new);
	if(ret)
	    krb5_warn(context, errno, "writing master key file");
    }

    hdb_free_master_key(context, mkey);
    return 0;
}
