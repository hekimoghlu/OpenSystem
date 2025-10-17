/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
/* $Id: result.h,v 1.9 2008/04/01 23:47:10 tbox Exp $ */

#ifndef DST_RESULT_H
#define DST_RESULT_H 1

/*! \file dst/result.h */

#include <isc/lang.h>
#include <isc/resultclass.h>

/*
 * Nothing in this file truly depends on <isc/result.h>, but the
 * DST result codes are considered to be publicly derived from
 * the ISC result codes, so including this file buys you the ISC_R_
 * namespace too.
 */
#include <isc/result.h>		/* Contractual promise. */

#define DST_R_UNSUPPORTEDALG		(ISC_RESULTCLASS_DST + 0)
#define DST_R_CRYPTOFAILURE		(ISC_RESULTCLASS_DST + 1)
/* compat */
#define DST_R_OPENSSLFAILURE		DST_R_CRYPTOFAILURE
#define DST_R_NOCRYPTO			(ISC_RESULTCLASS_DST + 2)
#define DST_R_NULLKEY			(ISC_RESULTCLASS_DST + 3)
#define DST_R_INVALIDPUBLICKEY		(ISC_RESULTCLASS_DST + 4)
#define DST_R_INVALIDPRIVATEKEY		(ISC_RESULTCLASS_DST + 5)
/* 6 is unused */
#define DST_R_WRITEERROR		(ISC_RESULTCLASS_DST + 7)
#define DST_R_INVALIDPARAM		(ISC_RESULTCLASS_DST + 8)
/* 9 is unused */
/* 10 is unused */
#define DST_R_SIGNFAILURE		(ISC_RESULTCLASS_DST + 11)
/* 12 is unused */
/* 13 is unused */
#define DST_R_VERIFYFAILURE		(ISC_RESULTCLASS_DST + 14)
#define DST_R_NOTPUBLICKEY		(ISC_RESULTCLASS_DST + 15)
#define DST_R_NOTPRIVATEKEY		(ISC_RESULTCLASS_DST + 16)
#define DST_R_KEYCANNOTCOMPUTESECRET	(ISC_RESULTCLASS_DST + 17)
#define DST_R_COMPUTESECRETFAILURE	(ISC_RESULTCLASS_DST + 18)
#define DST_R_NORANDOMNESS		(ISC_RESULTCLASS_DST + 19)
#define DST_R_BADKEYTYPE		(ISC_RESULTCLASS_DST + 20)
#define DST_R_NOENGINE			(ISC_RESULTCLASS_DST + 21)
#define DST_R_EXTERNALKEY		(ISC_RESULTCLASS_DST + 22)

#define DST_R_NRESULTS			23	/* Number of results */

ISC_LANG_BEGINDECLS

const char *
dst_result_totext(isc_result_t);

void
dst_result_register(void);

ISC_LANG_ENDDECLS

#endif /* DST_RESULT_H */
