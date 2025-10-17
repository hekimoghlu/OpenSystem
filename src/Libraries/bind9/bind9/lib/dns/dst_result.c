/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
/*%
 * Principal Author: Brian Wellington
 * $Id: dst_result.c,v 1.7 2008/04/01 23:47:10 tbox Exp $
 */

#include <config.h>

#include <isc/once.h>
#include <isc/util.h>

#include <dst/result.h>
#include <dst/lib.h>

static const char *text[DST_R_NRESULTS] = {
	"algorithm is unsupported",		/*%< 0 */
	"crypto failure",			/*%< 1 */
	"built with no crypto support",		/*%< 2 */
	"illegal operation for a null key",	/*%< 3 */
	"public key is invalid",		/*%< 4 */
	"private key is invalid",		/*%< 5 */
	"external key",				/*%< 6 */
	"error occurred writing key to disk",	/*%< 7 */
	"invalid algorithm specific parameter",	/*%< 8 */
	"UNUSED9",				/*%< 9 */
	"UNUSED10",				/*%< 10 */
	"sign failure",				/*%< 11 */
	"UNUSED12",				/*%< 12 */
	"UNUSED13",				/*%< 13 */
	"verify failure",			/*%< 14 */
	"not a public key",			/*%< 15 */
	"not a private key",			/*%< 16 */
	"not a key that can compute a secret",	/*%< 17 */
	"failure computing a shared secret",	/*%< 18 */
	"no randomness available",		/*%< 19 */
	"bad key type",				/*%< 20 */
	"no engine",				/*%< 21 */
	"illegal operation for an external key",/*%< 22 */
};

#define DST_RESULT_RESULTSET			2

static isc_once_t		once = ISC_ONCE_INIT;

static void
initialize_action(void) {
	isc_result_t result;

	result = isc_result_register(ISC_RESULTCLASS_DST, DST_R_NRESULTS,
				     text, dst_msgcat, DST_RESULT_RESULTSET);
	if (result != ISC_R_SUCCESS)
		UNEXPECTED_ERROR(__FILE__, __LINE__,
				 "isc_result_register() failed: %u", result);
}

static void
initialize(void) {
	dst_lib_initmsgcat();
	RUNTIME_CHECK(isc_once_do(&once, initialize_action) == ISC_R_SUCCESS);
}

const char *
dst_result_totext(isc_result_t result) {
	initialize();

	return (isc_result_totext(result));
}

void
dst_result_register(void) {
	initialize();
}

/*! \file */
