/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

/*
 * Test run functions, to be used with valgrind to detect memoryleaks.
 */

static void
check_log(void)
{
    int i;

    for (i = 0; i < 10; i++) {
	krb5_log_facility *logfacility;
	krb5_context context;
	krb5_error_code ret;

	ret = krb5_init_context(&context);
	if (ret)
	    errx (1, "krb5_init_context failed: %d", ret);

	krb5_initlog(context, "test-mem", &logfacility);
	krb5_addlog_dest(context, logfacility, "0/STDERR:");
	krb5_set_warn_dest(context, logfacility);

	krb5_free_context(context);
    }
}


int
main(int argc, char **argv)
{
    setprogname(argv[0]);

    check_log();

    return 0;
}
