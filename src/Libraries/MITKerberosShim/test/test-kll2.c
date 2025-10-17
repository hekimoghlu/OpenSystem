/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#include "mit-KerberosLogin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

int
main(int argc, char **argv)
{
    KLLoginOptions options;
    KLPrincipal princ;
    KLStatus ret;
    KLBoolean foundV5;
    char *user;
    char *password;
    char *buffer;

    if (argc != 3)
	errx(1, "argc != 2");

    user = argv[1];
    password = strdup(argv[2]);

    printf("test NULL argument\n");
    ret = KLCreatePrincipalFromString(NULL, kerberosVersion_V5, &princ);
    if (ret == 0)
	errx(1, "KLCreatePrincipalFromString: %d", ret);

    printf("create principal\n");
    ret = KLCreatePrincipalFromString(user,
				      kerberosVersion_V5, &princ);
    if (ret)
	errx(1, "KLCreatePrincipalFromString: %d", ret);

    printf("acquire cred\n");
    ret = KLAcquireNewInitialTicketsWithPassword(princ, NULL, password, NULL);
    if (ret)
	errx(1, "KLAcquireTicketsWithPassword1: %d", ret);

    printf("get valid tickets\n");
    ret = KLCacheHasValidTickets(princ, kerberosVersion_V5, &foundV5, NULL, NULL);
    if (ret)
	errx(1, "KLCacheHasValidTickets failed1");
    else if (!foundV5)
	errx(1, "found no valid tickets");

#if 0
    ret = KLAcquireNewInitialTickets(princ, NULL, NULL, NULL);
    if (ret)
	errx(1, "KLAcquireTickets: %d", ret);

    KLDestroyTickets(princ);

    printf("get valid tickets\n");
    ret = KLCacheHasValidTickets(princ, kerberosVersion_V5, &foundV5, NULL, NULL);
    if (ret)
	errx(1, "KLCacheHasValidTickets failed1 dead");
    else if (foundV5)
	errx(1, "found valid tickets!");
#endif

    KLCreateLoginOptions(&options);
    KLLoginOptionsSetRenewableLifetime(options, 3600 * 24 * 7);

    ret = KLAcquireNewInitialTicketsWithPassword(princ, options, password, NULL);
    if (ret)
	errx(1, "KLAcquireTicketsWithPassword2: %d", ret);

    KLDisposeLoginOptions(options);

    printf("get valid tickets\n");
    ret = KLCacheHasValidTickets(princ, kerberosVersion_V5, &foundV5, NULL, NULL);
    if (ret)
	errx(1, "KLCacheHasValidTickets failed");
    else if (!foundV5)
	errx(1, "found no valid tickets");

    printf("renew tickets\n");
    ret = KLRenewInitialTickets(princ, NULL, NULL, NULL);
    if (ret)
	errx(1, "KLRenewInitialTickets: %d", ret);

    printf("display string from princ\n");
    ret = KLGetDisplayStringFromPrincipal(princ, kerberosVersion_V5, &buffer);
    if (ret)
	errx(1, "KLGetDisplayStringFromPrincipal: %d", ret);
    free(buffer);

    printf("string from princ\n");
    ret = KLGetStringFromPrincipal(princ, kerberosVersion_V5, &buffer);
    if (ret)
	errx(1, "KLGetStringFromPrincipal: %d", ret);
    free(buffer);

    {
    	char *name;
	char *inst;
	char *realm;
        printf("triplet from princ\n");
        ret = KLGetTripletFromPrincipal(princ, &name, &inst, &realm);
        if (ret)
	    errx(1, "KLCancelAllDialogs: %d", ret);
	free(name);
	free(inst);
	free(realm);
    }


    printf("cancel dialogs\n");
    ret = KLCancelAllDialogs();
    if (ret)
	errx(1, "KLCancelAllDialogs: %d", ret);

    printf("dispose string\n");
    ret = KLDisposeString(password);
    if (ret)
	errx(1, "KLDisposeString: %d", ret);

    KLDestroyTickets(princ);
    KLDisposePrincipal(princ);

    return 0;
}
