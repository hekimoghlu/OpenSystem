/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#include "krb5_locl.h"
#include <err.h>

/*
 *
 */

static void
check_linear(krb5_context context,
	     const char *client_realm,
	     const char *server_realm,
	     const char *realm,
	     ...)
{
    unsigned int num_inrealms = 0, num_realms = 0, n;
    char **inrealms = NULL;
    char **realms = NULL;
    krb5_error_code ret;
    krb5_data tr;
    va_list va;

    krb5_data_zero(&tr);

    va_start(va, realm);

    while (realm) {
	inrealms = erealloc(inrealms, (num_inrealms + 2) * sizeof(inrealms[0]));
	inrealms[num_inrealms] = rk_UNCONST(realm);
	num_inrealms++;
	realm = va_arg(va, const char *);
    }
    if (inrealms)
	inrealms[num_inrealms] = NULL;

    ret = krb5_domain_x500_encode(inrealms, num_inrealms, &tr);
    if (ret)
	krb5_err(context, 1, ret, "krb5_domain_x500_encode");

    ret = krb5_domain_x500_decode(context, tr,
				  &realms, &num_realms,
				  client_realm, server_realm);
    if (ret)
	krb5_err(context, 1, ret, "krb5_domain_x500_decode");

    krb5_data_free(&tr);

    if (num_inrealms != num_realms)
	errx(1, "num_inrealms != num_realms");

    for(n = 0; n < num_realms; n++)
	free(realms[n]);
    free(realms);

    free(inrealms);
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context");


    check_linear(context, "KTH1.SE", "KTH1.SE", NULL);
    check_linear(context, "KTH1.SE", "KTH2.SE", NULL);
    check_linear(context, "KTH1.SE", "KTH3.SE", "KTH2.SE", NULL);
    check_linear(context, "KTH1.SE", "KTH4.SE", "KTH3.SE", "KTH2.SE", NULL);
    check_linear(context, "KTH1.SE", "KTH5.SE", "KTH4.SE", "KTH3.SE", "KTH2.SE", NULL);

    return 0;
}
