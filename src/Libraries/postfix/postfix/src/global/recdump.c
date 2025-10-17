/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <stdlib.h>

/* Utility library. */

#include <msg_vstream.h>

/* Global library. */

#include <record.h>
#include <rec_streamlf.h>
#include <rec_type.h>

int     main(int unused_argc, char **argv)
{
    VSTRING *buf = vstring_alloc(100);
    long    offset;
    int     type;

    msg_vstream_init(argv[0], VSTREAM_OUT);

    while (offset = vstream_ftell(VSTREAM_IN),
	   ((type = rec_get(VSTREAM_IN, buf, 0)) != REC_TYPE_EOF
	   && type != REC_TYPE_ERROR)) {
	vstream_fprintf(VSTREAM_OUT, "%15s|%4ld|%3ld|%s\n",
			rec_type_name(type), offset,
			(long) VSTRING_LEN(buf), vstring_str(buf));
    }
    vstream_fflush(VSTREAM_OUT);
    vstring_free(buf);
    exit(0);
}
