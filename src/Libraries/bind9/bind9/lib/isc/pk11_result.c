/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include <config.h>
#include <stddef.h>

#include <isc/once.h>
#include <isc/msgcat.h>
#include <isc/util.h>

#include <pk11/result.h>

LIBISC_EXTERNAL_DATA isc_msgcat_t *		pk11_msgcat = NULL;

static isc_once_t		msgcat_once = ISC_ONCE_INIT;

static const char *text[PK11_R_NRESULTS] = {
	"PKCS#11 initialization failed",		/*%< 0 */
	"no PKCS#11 provider",				/*%< 1 */
	"PKCS#11 provider has no random service",	/*%< 2 */
	"PKCS#11 provider has no digest service",	/*%< 3 */
	"PKCS#11 provider has no AES service",		/*%< 4 */
};

#define PK11_RESULT_RESULTSET			2

static isc_once_t		once = ISC_ONCE_INIT;

static void
open_msgcat(void) {
	isc_msgcat_open("libpk11.cat", &pk11_msgcat);
}

void
pk11_initmsgcat(void) {

	/*
	 * Initialize the PKCS#11 support's message catalog,
	 * pk11_msgcat, if it has not already been initialized.
	 */

	RUNTIME_CHECK(isc_once_do(&msgcat_once, open_msgcat) == ISC_R_SUCCESS);
}

static void
initialize_action(void) {
	isc_result_t result;

	result = isc_result_register(ISC_RESULTCLASS_PK11, PK11_R_NRESULTS,
				     text, pk11_msgcat, PK11_RESULT_RESULTSET);
	if (result != ISC_R_SUCCESS)
		UNEXPECTED_ERROR(__FILE__, __LINE__,
				 "isc_result_register() failed: %u", result);
}

static void
initialize(void) {
	pk11_initmsgcat();
	RUNTIME_CHECK(isc_once_do(&once, initialize_action) == ISC_R_SUCCESS);
}

const char *
pk11_result_totext(isc_result_t result) {
	initialize();

	return (isc_result_totext(result));
}

void
pk11_result_register(void) {
	initialize();
}
