/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
/* $Id$ */

#ifndef KADM5_PWCHECK_H
#define KADM5_PWCHECK_H 1


#define KADM5_PASSWD_VERSION_V0 0
#define KADM5_PASSWD_VERSION_V1 1

typedef const char* (*kadm5_passwd_quality_check_func_v0)(krb5_context,
							  krb5_principal,
							  krb5_data*);

/*
 * The 4th argument, is a tuning parameter for the quality check
 * function, the lib/caller will providing it for the password quality
 * module.
 */

typedef int
(*kadm5_passwd_quality_check_func)(krb5_context context,
				   krb5_principal principal,
				   krb5_data *password,
				   const char *tuning,
				   char *message,
				   size_t length);

struct kadm5_pw_policy_check_func {
    const char *name;
    kadm5_passwd_quality_check_func func;
};

struct kadm5_pw_policy_verifier {
    const char *name;
    int version;
    const char *vendor;
    const struct kadm5_pw_policy_check_func *funcs;
};

#endif /* KADM5_PWCHECK_H */
