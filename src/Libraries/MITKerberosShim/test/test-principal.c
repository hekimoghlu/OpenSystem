/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include <Kerberos/Kerberos.h>
#include <string.h>
#include <err.h>


int
main(int argc, char **argv)
{
    krb5_principal p1, p2;
    krb5_context context;
    krb5_error_code ret;

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context");

    ret = krb5_parse_name(context, "lha@OD.APPLE.COM", &p1);
    if (ret)
	errx(1, "krb5_parse_name");
    ret = krb5_parse_name(context, "foo@OD.APPLE.COM", &p2);
    if (ret)
	errx(1, "krb5_parse_name");

    ret = krb5_realm_compare(context, p1, p2);
    if (ret == 0)
	errx(1, "krb5_realm_compare");
	
    ret = krb5_principal_compare(context, p1, p2);
    if (ret != 0)
	errx(1, "krb5_principal_compare");

    ret = krb5_cc_end_seq_get(context, NULL, NULL);
    if (ret)
	errx(1, "krb5_cc_end_seq_get");

    krb5_free_context(context);

    return 0;
}        

