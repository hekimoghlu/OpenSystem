/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

static void
check_set_time(krb5_context context)
{
    krb5_error_code ret;
    krb5_timestamp sec;
    int32_t usec;
    struct timeval tv;
    int diff = 10;
    int diff2;

    gettimeofday(&tv, NULL);

    ret = krb5_set_real_time(context, tv.tv_sec + diff, tv.tv_usec);
    if (ret)
	krb5_err(context, 1, ret, "krb5_us_timeofday");

    ret = krb5_us_timeofday(context, &sec, &usec);
    if (ret)
	krb5_err(context, 1, ret, "krb5_us_timeofday");

    diff2 = abs(sec - tv.tv_sec);

    if (diff2 < 9 || diff > 11)
	krb5_errx(context, 1, "set time error: diff: %d",
		  abs(sec - tv.tv_sec));
}



int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context %d", ret);

    check_set_time(context);
    check_set_time(context);
    check_set_time(context);
    check_set_time(context);
    check_set_time(context);

    krb5_free_context(context);

    return 0;
}
