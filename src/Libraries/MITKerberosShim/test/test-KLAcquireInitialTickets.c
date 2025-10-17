/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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


#include <stdio.h>
#include <err.h>

#include <Kerberos/Kerberos.h>
#include <Kerberos/KerberosLogin.h>
#include <Kerberos/KerberosLoginPrivate.h>

#include <err.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static KLStatus
AcquireTGT(char *inusername)
{
    KLStatus	status;
    KLPromptMechanism prompt;
    char *credCacheName = NULL;
    KLPrincipal klprincipal = NULL;
	
    prompt = __KLPromptMechanism ();
    __KLSetPromptMechanism (klPromptMechanism_GUI);

    if (inusername && inusername[0]) {
	// user specified, acquire tickets for this user, and make it default
	status = KLCreatePrincipalFromString(inusername, kerberosVersion_V5, &klprincipal);
	if (status == klNoErr) {
	    status = KLAcquireInitialTickets(klprincipal, NULL, NULL, &credCacheName);
	    if (status == klNoErr)
		status = KLSetSystemDefaultCache(klprincipal);
	}
    }
    else {
	// no user specified, use default configuration
	status = KLAcquireInitialTickets(NULL, NULL, NULL, &credCacheName);
    }
		
    if(klprincipal != NULL)
	KLDisposePrincipal(klprincipal);
    if(credCacheName != NULL)
	KLDisposeString(credCacheName);
    __KLSetPromptMechanism (prompt);

    return status;
}

int
main(int argc, char **argv)
{
	if (argc != 2)
		errx(1, "argc != 2");
	AcquireTGT(argv[1]);
}

#pragma clang diagnostic pop
