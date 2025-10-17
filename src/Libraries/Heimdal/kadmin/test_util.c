/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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

krb5_context context;
void *kadm_handle;

struct {
    const char *str;
    int ret;
    time_t t;
} ts[] = {
    { "2006-12-22 18:09:00", 0, 1166810940 },
    { "2006-12-22", 0, 1166831999 },
    { "2006-12-22 23:59:59", 0, 1166831999 }
};

static int
test_time(void)
{
    int i, errors = 0;

    for (i = 0; i < sizeof(ts)/sizeof(ts[0]); i++) {
	time_t t;
	int ret;

	ret = str2time_t (ts[i].str, &t);
	if (ret != ts[i].ret) {
	    printf("%d: %d is wrong ret\n", i, ret);
	    errors++;
	}
	else if (t != ts[i].t) {
	    printf("%d: %d is wrong time\n", i, (int)t);
	    errors++;
	}
    }

    return errors;
}


int
main(int argc, char **argv)
{
    krb5_error_code ret;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = 0;
    ret += test_time();

    krb5_free_context(context);

    return ret;
}

