/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#if CONFIG_EXCLAVES

#include <stdint.h>
#include <mach/exclaves.h>
#include <mach/kern_return.h>

#include "kern/exclaves.tightbeam.h"

#include "exclaves_frame_mint.h"
#include "exclaves_resource.h"
#include "exclaves_debug.h"

/* -------------------------------------------------------------------------- */
#pragma mark Frame Mint

#define EXCLAVES_FRAME_MINT "com.apple.service.FrameMint"

static framemint_framemint_s frame_mint_client;

/*
 * Called as part of the populate call. As we can't cleanup tightbeam
 * connections it just sticks around. If we ever need to make any other calls to
 * FrameMint, having it a separate function makes that easier.
 */
static kern_return_t
exclaves_frame_mint_init(void)
{
	exclaves_id_t id = exclaves_service_lookup(EXCLAVES_DOMAIN_KERNEL,
	    EXCLAVES_FRAME_MINT);
	if (id == EXCLAVES_INVALID_ID) {
		return KERN_NOT_FOUND;
	}

	tb_endpoint_t ep = tb_endpoint_create_with_value(
		TB_TRANSPORT_TYPE_XNU, id, TB_ENDPOINT_OPTIONS_NONE);

	tb_error_t tb_result = framemint_framemint__init(&frame_mint_client, ep);

	if (tb_result != TB_ERROR_SUCCESS) {
		exclaves_debug_printf(show_errors,
		    "frame mint init: failure %u\n", tb_result);
		return KERN_FAILURE;
	}

	return KERN_SUCCESS;
}

kern_return_t
exclaves_frame_mint_populate(void)
{
	__block bool success = false;
	tb_error_t tb_result = TB_ERROR_SUCCESS;

	kern_return_t kr = exclaves_frame_mint_init();
	if (kr != KERN_SUCCESS) {
		return kr;
	}

	/* BEGIN IGNORE CODESTYLE */
	tb_result = framemint_framemint_populate(&frame_mint_client,
	    ^(framemint_framemint_populate__result_s result) {
		if (framemint_framemint_populate__result_get_success(&result)) {
			success = true;
			return;
		}

		framemint_frameminterror_s *error = NULL;
		error = framemint_framemint_populate__result_get_failure(&result);

		assert3p(error, !=, NULL);
		exclaves_debug_printf(show_errors,
		    "frame mint failure: failure %u\n", *error);
	});
	/* END IGNORE CODESTYLE */

	if (tb_result != TB_ERROR_SUCCESS || !success) {
		return KERN_FAILURE;
	}

	return KERN_SUCCESS;
}

#endif /* CONFIG_EXCLAVES */
