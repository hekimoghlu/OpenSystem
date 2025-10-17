/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
/* $Id: lib.c,v 1.16 2009/09/02 23:48:02 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include <isc/app.h>
#include <isc/lib.h>
#include <isc/mem.h>
#include <isc/msgs.h>
#include <isc/once.h>
#include <isc/print.h>
#include <isc/socket.h>
#include <isc/task.h>
#include <isc/timer.h>
#include <isc/util.h>

/***
 *** Globals
 ***/

LIBISC_EXTERNAL_DATA isc_msgcat_t *		isc_msgcat = NULL;


/***
 *** Private
 ***/

static isc_once_t		msgcat_once = ISC_ONCE_INIT;

/***
 *** Functions
 ***/

static void
open_msgcat(void) {
	isc_msgcat_open("libisc.cat", &isc_msgcat);
}

void
isc_lib_initmsgcat(void) {
	isc_result_t result;

	/*!
	 * Initialize the ISC library's message catalog, isc_msgcat, if it
	 * has not already been initialized.
	 */

	result = isc_once_do(&msgcat_once, open_msgcat);
	if (result != ISC_R_SUCCESS) {
		/*
		 * Normally we'd use RUNTIME_CHECK() or FATAL_ERROR(), but
		 * we can't do that here, since they might call us!
		 * (Note that the catalog might be open anyway, so we might
		 * as well try to  provide an internationalized message.)
		 */
		fprintf(stderr, "%s:%d: %s: isc_once_do() %s.\n",
			__FILE__, __LINE__,
			isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
				       ISC_MSG_FATALERROR, "fatal error"),
			isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
				       ISC_MSG_FAILED, "failed"));
		abort();
	}
}

static isc_once_t		register_once = ISC_ONCE_INIT;

static void
do_register(void) {
	isc_bind9 = ISC_FALSE;

	RUNTIME_CHECK(isc__mem_register() == ISC_R_SUCCESS);
	RUNTIME_CHECK(isc__app_register() == ISC_R_SUCCESS);
	RUNTIME_CHECK(isc__task_register() == ISC_R_SUCCESS);
	RUNTIME_CHECK(isc__socket_register() == ISC_R_SUCCESS);
	RUNTIME_CHECK(isc__timer_register() == ISC_R_SUCCESS);
}

void
isc_lib_register(void) {
	RUNTIME_CHECK(isc_once_do(&register_once, do_register)
		      == ISC_R_SUCCESS);
}
