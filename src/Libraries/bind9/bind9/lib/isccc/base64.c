/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
/* $Id: base64.c,v 1.8 2007/08/28 07:20:43 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/base64.h>
#include <isc/buffer.h>
#include <isc/region.h>
#include <isc/result.h>

#include <isccc/base64.h>
#include <isccc/result.h>
#include <isccc/util.h>

isc_result_t
isccc_base64_encode(isccc_region_t *source, int wordlength,
		  const char *wordbreak, isccc_region_t *target)
{
	isc_region_t sr;
	isc_buffer_t tb;
	isc_result_t result;

	sr.base = source->rstart;
	sr.length = (unsigned int)(source->rend - source->rstart);
	isc_buffer_init(&tb, target->rstart,
			(unsigned int)(target->rend - target->rstart));

	result = isc_base64_totext(&sr, wordlength, wordbreak, &tb);
	if (result != ISC_R_SUCCESS)
		return (result);
	source->rstart = source->rend;
	target->rstart = isc_buffer_used(&tb);
	return (ISC_R_SUCCESS);
}

isc_result_t
isccc_base64_decode(const char *cstr, isccc_region_t *target) {
	isc_buffer_t b;
	isc_result_t result;

	isc_buffer_init(&b, target->rstart,
			(unsigned int)(target->rend - target->rstart));
	result = isc_base64_decodestring(cstr, &b);
	if (result != ISC_R_SUCCESS)
		return (result);
	target->rstart = isc_buffer_used(&b);
	return (ISC_R_SUCCESS);
}
