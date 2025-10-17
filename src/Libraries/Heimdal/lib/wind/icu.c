/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#include <unicode/usprep.h>

int
wind_stringprep(const uint32_t *in, size_t in_len,
		uint32_t *out, size_t *out_len,
		wind_profile_flags flags)
{
    UErrorCode status = 0;
    UStringPrepProfile *profile;
    UStringPrepProfileType type;
    UChar *uin, *dest;
    int32_t len;
    size_t n;

    if (in_len > UINT_MAX / sizeof(in[0]) || (*out_len) > UINT_MAX / sizeof(out[0]))
	return EINVAL;

    if (flags & WIND_PROFILE_SASL)
	type = USPREP_RFC4013_SASLPREP;
    else
	return EINVAL;

    /*
     * Should cache profile
     */

    profile = usprep_openByType(type, &status);
    if (profile == NULL)
	return ENOENT;

    uin = malloc(in_len * sizeof(uin[0]));
    dest = malloc(*out_len * sizeof(dest[0]));
    if (uin == NULL || dest == NULL) {
	free(uin);
	free(dest);
	usprep_close(profile);
	return ENOMEM;
    }
    
    /* ucs42ucs2 - don't care about surogates */
    for (n = 0; n < in_len; n++)
	uin[n] = in[n];

    status = 0;

    len = usprep_prepare(profile, uin, (int32_t)in_len, dest, (int32_t)*out_len,
			 USPREP_DEFAULT, NULL, &status);
    
    if (len < 0 || status) {
	free(dest);
	free(uin);
	return EINVAL;
    }

    for (n = 0; n < (size_t)len; n++)
	out[n] = dest[n];

    *out_len = len;

    free(dest);
    free(uin);

    return 0;
}
