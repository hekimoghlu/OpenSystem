/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

/* Utility library. */

#include <msg.h>
#include <vstream.h>

/* Global library. */

#include "record.h"
#include "rec_type.h"
#include "deliver_completed.h"

/* deliver_completed - handle per-recipient delivery completion */

void    deliver_completed(VSTREAM *stream, long offset)
{
    const char *myname = "deliver_completed";

    if (offset == -1)
	return;

    if (offset <= 0)
	msg_panic("%s: bad offset %ld", myname, offset);

    if (rec_put_type(stream, REC_TYPE_DONE, offset) < 0
	|| vstream_fflush(stream))
	msg_fatal("update queue file %s: %m", VSTREAM_PATH(stream));
}
