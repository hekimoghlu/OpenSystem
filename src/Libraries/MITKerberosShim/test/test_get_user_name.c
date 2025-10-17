/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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


#include <Kerberos/KerberosLogin.h>
#include <stdio.h>
#include <err.h>

int
main(int argc, char **argv)
{
    int		error;
    char	*outName = NULL;
    char	*name, *instance, *realm;
    KLBoolean	foundTickets = 0;
    char	*cacheName;
    KLPrincipal	principal;
    uint32_t	version;
    
    error = KLCacheHasValidTickets(NULL, kerberosVersion_V5,
				   &foundTickets, NULL, NULL);
    if (error)
	errx(1, "no valid ticket");

    error = KLCacheHasValidTickets(NULL, kerberosVersion_V5,
				   &foundTickets, &principal, &cacheName);
    if (error)
	errx(1, "no valid ticket");
    
    error = KLGetTripletFromPrincipal (principal, &name, &instance, &realm);
    KLDisposePrincipal (principal);
    if (error)
	errx(1, "failed to parse principal");
    
    printf("name: %s instance: %s realm: %s cacheName: %s\n", name, instance, realm, cacheName);
    
    KLDisposeString (name);
    KLDisposeString (instance);
    KLDisposeString (realm);
    KLDisposeString (cacheName);

    return 0;
}
