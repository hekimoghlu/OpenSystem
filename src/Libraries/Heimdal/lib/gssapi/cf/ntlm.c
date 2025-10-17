/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#include "mech_locl.h"
#include <heim_threads.h>

#include <Security/Security.h>
#include "krb5.h"
#include "heimcred.h"

/*
 *
 */

bool
GSSCheckNTLMReflection(uint8_t challenge[8])
{
    krb5_boolean found_reflection = false;
#ifdef HAVE_KCM
    static krb5_context context;
    static dispatch_once_t once;
    krb5_error_code ret;

    dispatch_once(&once, ^{
	    krb5_init_context(&context);
	});

    if (context == NULL) /* fail open for now */
	return false;

    ret = krb5_kcm_check_ntlm_challenge(context, challenge, &found_reflection);
    if (ret)
	return false; /* fail open for now */
#else
    found_reflection = HeimCredCheckNTLMChallenge(challenge);
#endif

    return found_reflection;
}
