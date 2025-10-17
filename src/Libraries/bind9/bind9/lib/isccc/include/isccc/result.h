/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
/* $Id: result.h,v 1.12 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_RESULT_H
#define ISCCC_RESULT_H 1

/*! \file isccc/result.h */

#include <isc/lang.h>
#include <isc/resultclass.h>
#include <isc/result.h>

#include <isccc/types.h>

/*% Unknown Version */
#define ISCCC_R_UNKNOWNVERSION		(ISC_RESULTCLASS_ISCCC + 0)
/*% Syntax Error */
#define ISCCC_R_SYNTAX			(ISC_RESULTCLASS_ISCCC + 1)
/*% Bad Authorization */
#define ISCCC_R_BADAUTH			(ISC_RESULTCLASS_ISCCC + 2)
/*% Expired */
#define ISCCC_R_EXPIRED			(ISC_RESULTCLASS_ISCCC + 3)
/*% Clock Skew */
#define ISCCC_R_CLOCKSKEW		(ISC_RESULTCLASS_ISCCC + 4)
/*% Duplicate */
#define ISCCC_R_DUPLICATE		(ISC_RESULTCLASS_ISCCC + 5)

#define ISCCC_R_NRESULTS 		6	/*%< Number of results */

ISC_LANG_BEGINDECLS

const char *
isccc_result_totext(isc_result_t result);
/*%
 * Convert a isccc_result_t into a string message describing the result.
 */

void
isccc_result_register(void);

ISC_LANG_ENDDECLS

#endif /* ISCCC_RESULT_H */
