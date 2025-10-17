/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
rename_entry(void *opt, int argc, char **argv)
{
    krb5_error_code ret;
    krb5_principal princ1, princ2;

    ret = krb5_parse_name(context, argv[0], &princ1);
    if(ret){
	krb5_warn(context, ret, "krb5_parse_name(%s)", argv[0]);
	return ret != 0;
    }
    ret = krb5_parse_name(context, argv[1], &princ2);
    if(ret){
	krb5_free_principal(context, princ1);
	krb5_warn(context, ret, "krb5_parse_name(%s)", argv[1]);
	return ret != 0;
    }
    ret = kadm5_rename_principal(kadm_handle, princ1, princ2);
    if(ret)
	krb5_warn(context, ret, "rename");
    krb5_free_principal(context, princ1);
    krb5_free_principal(context, princ2);
    return ret != 0;
}

