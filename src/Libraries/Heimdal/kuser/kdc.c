/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "kuser_locl.h"
#include <config.h>
#include "kcc-commands.h"

static const struct { 
    char *name;
    int type;
} types[] = {
    { "kdc", KRB5_KRBHST_KDC },
    { "kadmin", KRB5_KRBHST_ADMIN },
    { "changepw", KRB5_KRBHST_CHANGEPW },
    { "krb524", KRB5_KRBHST_KRB524 },
    { "kkdcp", KRB5_KRBHST_KKDCP }
};

int
kdc(struct kdc_options *opt, int argc, char **argv)
{
    krb5_krbhst_handle handle;
    int type = KRB5_KRBHST_KDC;
    char host[MAXHOSTNAMELEN];
    krb5_error_code ret;
    int first_realm = 1;
    krb5_uuid uuid;
    size_t n;
    int i;
    
    if (argc == 0) {
	printf("give at least on realm\n");
	return 1;
    }

    if (opt->type_string) {

	for (n = 0; n < sizeof(types)/sizeof(types[0]); n++) {
	    if (strcasecmp(types[n].name, opt->type_string) == 0) {
		type = types[n].type;
		break;
	    }
	}
	if (n == sizeof(types)/sizeof(types[0])) {
	    printf("unknown type: %s\nAvailaile types are: \n", opt->type_string);
	    for (n = 0; n < sizeof(types)/sizeof(types[0]); n++)
		printf("%s ", types[n].name);
	    printf("\n");
	    return 1;
	}
    }

    if (opt->uuid_string) {
	if (uuid_parse(opt->uuid_string, uuid) != 0)
	    errx(1, "failed tp parse `%s` as a uuid", opt->uuid_string);
    }

    if (opt->json_flag)
	printf("{");

    for (i = 0; i < argc; i++) {
	const char *realm = argv[i];

	ret = krb5_krbhst_init(kcc_context, realm, type, &handle);
	if (ret) {
	    krb5_warn(kcc_context, ret, "krb5_krbhst_init");
	    return 1;
	}

	if (opt->uuid_string)
	    krb5_krbhst_set_delgated_uuid(kcc_context, handle, uuid);

	if (opt->json_flag) {
	    int first = 1;

	    printf("%s\n\t\"%s\" = [ ", first_realm ? "" : ",", realm);
	    first_realm = 0;

	    while(krb5_krbhst_next_as_string(kcc_context, handle, host, sizeof(host)) == 0) {
		printf("%s\n\t\t\"%s\"", first ? "" : ",", host);
		first = 0;
	    }

	    printf("\n\t]");
	} else {
	    krb5_krbhst_info *hi = NULL;

	    printf("[realms]\n");
	    printf("\t%s = {\n", realm);

	    while ((ret = krb5_krbhst_next(kcc_context, handle, &hi)) == 0) {

		ret = krb5_krbhst_format_string(kcc_context, hi, host, sizeof(host));
		if (ret)
		    krb5_warn(kcc_context, ret, "krb5_krbhst_format_string");

		printf("\t\tkdc = %s  # source %s\n", host, hi->source);
	    }
	    if (ret != KRB5_KDC_UNREACH)
		krb5_err(kcc_context, 1, ret, "Failed before getting to end of kdc list");
	    
	    printf("\t}\n");
	}

	krb5_krbhst_free(kcc_context, handle);
    }

    if (opt->json_flag)
	printf("\n}\n");

    return 0;
}
