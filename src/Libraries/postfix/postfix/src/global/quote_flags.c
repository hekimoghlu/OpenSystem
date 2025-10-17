/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
/*
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <name_mask.h>

 /*
  * Global library.
  */
#include <quote_flags.h>

static const NAME_MASK quote_flags_table[] = {
    "8bitclean", QUOTE_FLAG_8BITCLEAN,
    "expose_at", QUOTE_FLAG_EXPOSE_AT,
    "append", QUOTE_FLAG_APPEND,
    "bare_localpart", QUOTE_FLAG_BARE_LOCALPART,
    0,
};

/* quote_flags_from_string - symbolic quote flags to internal form */

int     quote_flags_from_string(const char *quote_flags_string)
{
    return (name_mask_delim_opt("quote_flags_from_string", quote_flags_table,
				quote_flags_string, "|",
				NAME_MASK_WARN | NAME_MASK_ANY_CASE));
}

/* quote_flags_to_string - internal form to symbolic quote flags */

const char *quote_flags_to_string(VSTRING *res_buf, int quote_flags_mask)
{
    static VSTRING *my_buf;

    if (res_buf == 0 && (res_buf = my_buf) == 0)
	res_buf = my_buf = vstring_alloc(20);
    return (str_name_mask_opt(res_buf, "quote_flags_to_string",
			      quote_flags_table, quote_flags_mask,
			      NAME_MASK_WARN | NAME_MASK_PIPE));
}
