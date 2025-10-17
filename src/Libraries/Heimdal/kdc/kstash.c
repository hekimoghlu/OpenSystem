/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "headers.h"

krb5_context context;

static char *keyfile;
static int convert_flag;
static int help_flag;
static int version_flag;

static int master_key_fd = -1;
static int random_key_flag;

static const char *enctype_str = "des3-cbc-sha1";

static struct getargs args[] = {
    { "enctype", 'e', arg_string, rk_UNCONST(&enctype_str), "encryption type",
	NULL },
    { "key-file", 'k', arg_string, &keyfile, "master key file", "file" },
    { "convert-file", 0, arg_flag, &convert_flag,
      "just convert keyfile to new format", NULL },
    { "master-key-fd", 0, arg_integer, &master_key_fd,
      "filedescriptor to read passphrase from", "fd" },
    { "random-key", 0, arg_flag, &random_key_flag,
	"generate a random master key", NULL },
    { "help", 'h', arg_flag, &help_flag, NULL, NULL },
    { "version", 0, arg_flag, &version_flag, NULL, NULL }
};

int num_args = sizeof(args) / sizeof(args[0]);

int
main(int argc, char **argv)
{
    char buf[1024];
    krb5_error_code ret;

    krb5_enctype enctype;

    hdb_master_key mkey;

    krb5_program_setup(&context, argc, argv, args, num_args, NULL);

    if(help_flag)
	krb5_std_usage(0, args, num_args);
    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    if (master_key_fd != -1 && random_key_flag)
	krb5_errx(context, 1, "random-key and master-key-fd "
		  "is mutual exclusive");

    if (keyfile == NULL)
	asprintf(&keyfile, "%s/m-key", hdb_db_dir(context));

    ret = krb5_string_to_enctype(context, enctype_str, &enctype);
    if(ret)
	krb5_err(context, 1, ret, "krb5_string_to_enctype");

    ret = hdb_read_master_key(context, keyfile, &mkey);
    if(ret && ret != ENOENT)
	krb5_err(context, 1, ret, "reading master key from %s", keyfile);

    if (convert_flag) {
	if (ret)
	    krb5_err(context, 1, ret, "reading master key from %s", keyfile);
    } else {
	krb5_keyblock key;
	krb5_salt salt;
	salt.salttype = KRB5_PW_SALT;
	/* XXX better value? */
	salt.saltvalue.data = NULL;
	salt.saltvalue.length = 0;
	if (random_key_flag) {
	    ret = krb5_generate_random_keyblock(context, enctype, &key);
	    if (ret)
		krb5_err(context, 1, ret, "krb5_generate_random_keyblock");

	} else {
	    if(master_key_fd != -1) {
		ssize_t n;
		n = read(master_key_fd, buf, sizeof(buf));
		if(n <= 0)
		    krb5_err(context, 1, errno, "failed to read passphrase");
		buf[n] = '\0';
		buf[strcspn(buf, "\r\n")] = '\0';

	    } else {
		if(UI_UTIL_read_pw_string(buf, sizeof(buf), "Master key: ", 1))
		    exit(1);
	    }
	    krb5_string_to_key_salt(context, enctype, buf, salt, &key);
	}
	ret = hdb_add_master_key(context, &key, &mkey);

	krb5_free_keyblock_contents(context, &key);

    }

    {
	char *new, *old;
	asprintf(&old, "%s.old", keyfile);
	asprintf(&new, "%s.new", keyfile);
	if(unlink(new) < 0 && errno != ENOENT) {
	    ret = errno;
	    goto out;
	}
	krb5_warnx(context, "writing key to `%s'", keyfile);
	ret = hdb_write_master_key(context, new, mkey);
	if(ret)
	    unlink(new);
	else {
#ifndef NO_POSIX_LINKS
	    unlink(old);
	    if(link(keyfile, old) < 0 && errno != ENOENT) {
		ret = errno;
		unlink(new);
	    } else {
#endif
		if(rename(new, keyfile) < 0) {
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

    exit(ret != 0);
}
