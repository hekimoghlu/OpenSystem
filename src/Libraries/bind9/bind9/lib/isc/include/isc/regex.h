/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#ifndef ISC_REGEX_H
#define ISC_REGEX_H 1

/*! \file isc/regex.h */

#include <isc/types.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

int
isc_regex_validate(const char *expression);
/*%<
 * Check a regular expression for syntactic correctness.
 *
 * Returns:
 *\li	 -1 on error.
 *\li	 the number of groups in the expression.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_REGEX_H */
