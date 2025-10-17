/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#include <vstring.h>
#include <vstream.h>

/* Global library. */

#include <record.h>
#include <rec_type.h>
#include <rec_streamlf.h>

/* rec_streamlf_get - read record from stream-lf file */

int     rec_streamlf_get(VSTREAM *stream, VSTRING *buf, int maxlen)
{
    int     n = maxlen;
    int     ch;

    /*
     * If this one character ar a time code proves to be a performance
     * bottleneck, switch to block search (memchr()) and to block move
     * (memcpy()) operations.
     */
    VSTRING_RESET(buf);
    while (n-- > 0) {
	if ((ch = VSTREAM_GETC(stream)) == VSTREAM_EOF)
	    return (VSTRING_LEN(buf) > 0 ? REC_TYPE_CONT : REC_TYPE_EOF);
	if (ch == '\n') {
	    VSTRING_TERMINATE(buf);
	    return (REC_TYPE_NORM);
	}
	VSTRING_ADDCH(buf, ch);
    }
    VSTRING_TERMINATE(buf);
    return (REC_TYPE_CONT);
}

/* rec_streamlf_put - write record to stream-lf file */

int     rec_streamlf_put(VSTREAM *stream, int type, const char *data, int len)
{
    if (len > 0)
	(void) vstream_fwrite(stream, data, len);
    if (type == REC_TYPE_NORM)
	(void) VSTREAM_PUTC('\n', stream);
    return (vstream_ferror(stream) ? REC_TYPE_EOF : type);
}
