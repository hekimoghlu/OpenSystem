/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

int
password_quality(void *opt, int argc, char **argv)
{
    krb5_error_code ret;
    krb5_principal principal;
    krb5_data pw_data;
    const char *s;

    ret = krb5_parse_name(context, argv[0], &principal);
    if(ret){
	krb5_warn(context, ret, "krb5_parse_name(%s)", argv[0]);
	return 0;
    }
    pw_data.data = argv[1];
    pw_data.length = strlen(argv[1]);

    s = kadm5_check_password_quality (context, principal, &pw_data);
    if (s)
	krb5_warnx(context, "kadm5_check_password_quality: %s", s);

    krb5_free_principal(context, principal);

    return 0;
}
