/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#ifndef PK11_RESULT_H
#define PK11_RESULT_H 1

/*! \file pk11/result.h */

#include <isc/lang.h>
#include <isc/resultclass.h>

/*
 * Nothing in this file truly depends on <isc/result.h>, but the
 * PK11 result codes are considered to be publicly derived from
 * the ISC result codes, so including this file buys you the ISC_R_
 * namespace too.
 */
#include <isc/result.h>		/* Contractual promise. */

#define PK11_R_INITFAILED		(ISC_RESULTCLASS_PK11 + 0)
#define PK11_R_NOPROVIDER		(ISC_RESULTCLASS_PK11 + 1)
#define PK11_R_NORANDOMSERVICE		(ISC_RESULTCLASS_PK11 + 2)
#define PK11_R_NODIGESTSERVICE		(ISC_RESULTCLASS_PK11 + 3)
#define PK11_R_NOAESSERVICE		(ISC_RESULTCLASS_PK11 + 4)

#define PK11_R_NRESULTS			5	/* Number of results */

ISC_LANG_BEGINDECLS

LIBISC_EXTERNAL_DATA extern isc_msgcat_t *pk11_msgcat;

void
pk11_initmsgcat(void);

const char *
pk11_result_totext(isc_result_t);

void
pk11_result_register(void);

ISC_LANG_ENDDECLS

#endif /* PK11_RESULT_H */
