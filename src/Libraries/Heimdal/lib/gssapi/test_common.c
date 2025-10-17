/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "config.h"

#include <stdio.h>
#include <gssapi.h>
#include <err.h>
#include "test_common.h"

char *
gssapi_err(OM_uint32 maj_stat, OM_uint32 min_stat, gss_OID mech)
{
	OM_uint32 junk;
	gss_buffer_desc maj_error_message;
	gss_buffer_desc min_error_message;
	OM_uint32 msg_ctx;

	char *ret = NULL;

	maj_error_message.length = 0;
	maj_error_message.value = NULL;
	min_error_message.length = 0;
	min_error_message.value = NULL;
	
	msg_ctx = 0;
	(void)gss_display_status(&junk, maj_stat,
				 GSS_C_GSS_CODE,
				 mech, &msg_ctx, &maj_error_message);
	msg_ctx = 0;
	(void)gss_display_status(&junk, min_stat,
				 GSS_C_MECH_CODE,
				 mech, &msg_ctx, &min_error_message);
	if (asprintf(&ret, "gss-code: %lu %.*s -- mech-code: %lu %.*s",
		     (unsigned long)maj_stat,
		     (int)maj_error_message.length,
		     (char *)maj_error_message.value,
		     (unsigned long)min_stat,
		     (int)min_error_message.length,
		     (char *)min_error_message.value) < 0 || ret == NULL)
	    errx(1, "malloc");

	gss_release_buffer(&junk, &maj_error_message);
	gss_release_buffer(&junk, &min_error_message);

	return ret;
}

