/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
/* $Id: assertions.c,v 1.26 2009/09/29 15:06:07 fdupont Exp $ */

/*! \file */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include <isc/assertions.h>
#include <isc/backtrace.h>
#include <isc/msgs.h>
#include <isc/print.h>
#include <isc/result.h>

/*
 * The maximum number of stack frames to dump on assertion failure.
 */
#ifndef BACKTRACE_MAXFRAME
#define BACKTRACE_MAXFRAME 128
#endif

/*%
 * Forward.
 */
static void
default_callback(const char *, int, isc_assertiontype_t, const char *);

static isc_assertioncallback_t isc_assertion_failed_cb = default_callback;

/*%
 * Public.
 */

/*% assertion failed handler */
/* coverity[+kill] */
void
isc_assertion_failed(const char *file, int line, isc_assertiontype_t type,
		     const char *cond)
{
	isc_assertion_failed_cb(file, line, type, cond);
	abort();
	/* NOTREACHED */
}

/*% Set callback. */
void
isc_assertion_setcallback(isc_assertioncallback_t cb) {
	if (cb == NULL)
		isc_assertion_failed_cb = default_callback;
	else
		isc_assertion_failed_cb = cb;
}

/*% Type to Text */
const char *
isc_assertion_typetotext(isc_assertiontype_t type) {
	const char *result;

	/*
	 * These strings have purposefully not been internationalized
	 * because they are considered to essentially be keywords of
	 * the ISC development environment.
	 */
	switch (type) {
	case isc_assertiontype_require:
		result = "REQUIRE";
		break;
	case isc_assertiontype_ensure:
		result = "ENSURE";
		break;
	case isc_assertiontype_insist:
		result = "INSIST";
		break;
	case isc_assertiontype_invariant:
		result = "INVARIANT";
		break;
	default:
		result = NULL;
	}
	return (result);
}

/*
 * Private.
 */

static void
default_callback(const char *file, int line, isc_assertiontype_t type,
		 const char *cond)
{
	void *tracebuf[BACKTRACE_MAXFRAME];
	int i, nframes;
	const char *logsuffix = ".";
	const char *fname;
	isc_result_t result;

	result = isc_backtrace_gettrace(tracebuf, BACKTRACE_MAXFRAME, &nframes);
		if (result == ISC_R_SUCCESS && nframes > 0)
			logsuffix = ", back trace";

	fprintf(stderr, "%s:%d: %s(%s) %s%s\n",
		file, line, isc_assertion_typetotext(type), cond,
		isc_msgcat_get(isc_msgcat, ISC_MSGSET_GENERAL,
			       ISC_MSG_FAILED, "failed"), logsuffix);
	if (result == ISC_R_SUCCESS) {
		for (i = 0; i < nframes; i++) {
			unsigned long offset;

			fname = NULL;
			result = isc_backtrace_getsymbol(tracebuf[i], &fname,
							 &offset);
			if (result == ISC_R_SUCCESS) {
				fprintf(stderr, "#%d %p in %s()+0x%lx\n", i,
					tracebuf[i], fname, offset);
			} else {
				fprintf(stderr, "#%d %p in ??\n", i,
					tracebuf[i]);
			}
		}
	}
	fflush(stderr);
}
