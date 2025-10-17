/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "gsskrb5_locl.h"

int
main(int argc, char **argv)
{
    OM_uint32 minor_status, maj_stat;
    gss_buffer_desc data;
    int ret;

    maj_stat = gss_oid_to_str(&minor_status, GSS_KRB5_MECHANISM, &data);
    if (GSS_ERROR(maj_stat))
	errx(1, "gss_oid_to_str failed");
    ret = strncmp(data.value, "1 2 840 113554 1 2 2", data.length);
    gss_release_buffer(&maj_stat, &data);
    if (ret)
	return 1;
    return 0;
}
