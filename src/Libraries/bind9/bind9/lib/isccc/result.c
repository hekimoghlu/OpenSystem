/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
/* $Id: result.c,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/once.h>
#include <isc/util.h>

#include <isccc/result.h>
#include <isccc/lib.h>

static const char *text[ISCCC_R_NRESULTS] = {
	"unknown version",			/* 1 */
	"syntax error",				/* 2 */
	"bad auth",				/* 3 */
	"expired",				/* 4 */
	"clock skew",				/* 5 */
	"duplicate"				/* 6 */
};

#define ISCCC_RESULT_RESULTSET			2

static isc_once_t		once = ISC_ONCE_INIT;

static void
initialize_action(void) {
	isc_result_t result;

	result = isc_result_register(ISC_RESULTCLASS_ISCCC, ISCCC_R_NRESULTS,
				     text, isccc_msgcat,
				     ISCCC_RESULT_RESULTSET);
	if (result != ISC_R_SUCCESS)
		UNEXPECTED_ERROR(__FILE__, __LINE__,
				 "isc_result_register() failed: %u", result);
}

static void
initialize(void) {
	isccc_lib_initmsgcat();
	RUNTIME_CHECK(isc_once_do(&once, initialize_action) == ISC_R_SUCCESS);
}

const char *
isccc_result_totext(isc_result_t result) {
	initialize();

	return (isc_result_totext(result));
}

void
isccc_result_register(void) {
	initialize();
}
