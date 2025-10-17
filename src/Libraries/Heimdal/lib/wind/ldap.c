/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "windlocl.h"
#include <assert.h>

static int
put_char(uint32_t *out, size_t *o, uint32_t c, size_t out_len)
{
    if (*o >= out_len)
	return 1;
    out[*o] = c;
    (*o)++;
    return 0;
}

int
_wind_ldap_case_exact_attribute(const uint32_t *tmp,
				size_t olen,
				uint32_t *out,
				size_t *out_len)
{
    size_t o = 0, i = 0;

    if (olen == 0) {
	*out_len = 0;
	return 0;
    }

    if (put_char(out, &o, 0x20, *out_len))
	return WIND_ERR_OVERRUN;
    while(i < olen && tmp[i] == 0x20) /* skip initial spaces */
	i++;

    while (i < olen) {
	if (tmp[i] == 0x20) {
	    if (put_char(out, &o, 0x20, *out_len) ||
		put_char(out, &o, 0x20, *out_len))
		return WIND_ERR_OVERRUN;
	    while(i < olen && tmp[i] == 0x20) /* skip middle spaces */
		i++;
	} else {
	    if (put_char(out, &o, tmp[i++], *out_len))
		return WIND_ERR_OVERRUN;
	}
    }
    assert(o > 0);

    /* only one spaces at the end */
    if (o == 1 && out[0] == 0x20)
	o = 0;
    else if (out[o - 1] == 0x20) {
	if (out[o - 2] == 0x20)
	    o--;
    } else
	put_char(out, &o, 0x20, *out_len);

    *out_len = o;

    return 0;
}
