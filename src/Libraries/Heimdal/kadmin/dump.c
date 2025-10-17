/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#ifdef __APPLE__
#include <HeimODAdmin.h>
#endif

extern int local_flag;

#ifdef __APPLE__

static krb5_error_code
od_dump_entry(krb5_context kcontext, HDB *db, hdb_entry_ex *entry, void *data)
{
    CFErrorRef error = NULL;
    CFDictionaryRef dict;
    CFStringRef fn, uuidstr;
    CFUUIDRef uuid;
    CFURLRef url;

    dict = HeimODDumpHdbEntry(&entry->entry, &error);
    if (dict == NULL) {
	if (error)
	    CFRelease(error);
	return 0;
    }

    uuid = CFUUIDCreate(NULL);
    if (uuid == NULL) {
	krb5_warnx(kcontext, "out of memory");
        CFRelease(dict);
	return 0;
    }
    
    uuidstr = CFUUIDCreateString(NULL, uuid);
    CFRelease(uuid);
    if (uuidstr == NULL) {
	krb5_warnx(kcontext, "out of memory");
        CFRelease(dict);
	return 0;
    }

    fn = CFStringCreateWithFormat(NULL, NULL, CFSTR("%s/%@.plist"), (char *)data, uuidstr);
    CFRelease(uuidstr);
    if (fn == NULL) {
	krb5_warnx(kcontext, "out of memory");
        CFRelease(dict);
	return 0;
    }

    url = CFURLCreateWithFileSystemPath(NULL, fn,  kCFURLPOSIXPathStyle, false);
    CFRelease(fn);
    if (url == NULL) {
	krb5_warnx(kcontext, "out of memory");
        CFRelease(dict);
	return 0;
    }

    CFDataRef xmldata = CFPropertyListCreateData(NULL, dict, kCFPropertyListXMLFormat_v1_0, 0, NULL);
    CFRelease(dict);
    if (xmldata == NULL) {
	CFRelease(url);
	krb5_warnx(kcontext, "out of memory");
	return 0;
    }

    CFWriteStreamRef stream = CFWriteStreamCreateWithFile(NULL, url);
    if (stream) {
	if (CFWriteStreamOpen(stream))
	    CFWriteStreamWrite(stream, CFDataGetBytePtr(xmldata), CFDataGetLength(xmldata));
	CFWriteStreamClose(stream);
	CFRelease(stream);
    }

    CFRelease(url);
    CFRelease(xmldata);

    return 0;
}
#endif

int
dump(struct dump_options *opt, int argc, char **argv)
{
    krb5_error_code (*func)(krb5_context, HDB *, hdb_entry_ex *, void *);
    krb5_error_code ret;
    void *arg;
    const char *format = "heimdal";
    FILE *f = NULL;

    if (opt->format_string)
	format = opt->format_string;

    if (strcasecmp(format, "heimdal") == 0) {
	func = hdb_print_entry;

	if (argc == 0) {
	    arg = stdout;
	} else {
	    arg = f = fopen(argv[0], "w");
	    if (f == NULL) {
		krb5_warn(context, errno, "failed to open %s", argv[0]);
		return 0;
	    }
	}
#ifdef __APPLE__
    } else if (strcasecmp(format, "od") == 0) {
	
	func = od_dump_entry;
	if (argc == 0)
	    arg = rk_UNCONST(".");
	else
	    arg = argv[0];
#endif
    } else {
	krb5_warnx(context, "unknown dump format: %s", format);
	return 0;
    }

    if (opt->mit_dump_file_string) {
	ret = hdb_mit_dump(context, opt->mit_dump_file_string,
			   func, arg);
	if (ret)
	    krb5_warn(context, ret, "hdb_mit_dump");

    } else {
	HDB *db = NULL;

	if (!local_flag) {
	    krb5_warnx(context, "od-dump is only available in local (-l) mode");
	    return 0;
	}

	db = _kadm5_s_get_db(kadm_handle);

	ret = db->hdb_open(context, db, O_RDONLY, 0600);
	if (ret) {
	    krb5_warn(context, ret, "hdb_open");
	    goto out;
	}

	ret = hdb_foreach(context, db, opt->decrypt_flag ? HDB_F_DECRYPT : 0,
			  func, arg);
	if (ret)
	    krb5_warn(context, ret, "hdb_foreach");

	db->hdb_close(context, db);
    }
    if (f)
	fclose(f);
out:
    return ret != 0;
}

int
od_dump(struct od_dump_options *opt, int argc, char **argv)
{
    struct dump_options dumpopt;

    memset(&dumpopt, 0, sizeof(dumpopt));
    dumpopt.decrypt_flag = opt->decrypt_flag;
    dumpopt.format_string = "od";

    return dump(&dumpopt, argc, argv);
}
