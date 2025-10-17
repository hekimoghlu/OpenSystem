/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

static int
do_del_entry(krb5_principal principal, void *data)
{
    return kadm5_delete_principal(kadm_handle, principal);
}

int
del_entry(void *opt, int argc, char **argv)
{
    int i;
    krb5_error_code ret = 0;

    for(i = 0; i < argc; i++) {
	ret = foreach_principal(argv[i], do_del_entry, "del", NULL);
	if (ret)
	    break;
    }
    return ret != 0;
}
