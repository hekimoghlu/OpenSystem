/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#ifndef UNIX_ERRNO2RESULT_H
#define UNIX_ERRNO2RESULT_H 1

/*! \file */

/* XXXDCL this should be moved to lib/isc/include/isc/errno2result.h. */

#include <errno.h>		/* Provides errno. */

#include <isc/lang.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

#define isc__errno2result(x) isc___errno2result(x, ISC_TRUE, __FILE__, __LINE__)

isc_result_t
isc___errno2result(int posixerrno, isc_boolean_t dolog,
		   const char *file, unsigned int line);

ISC_LANG_ENDDECLS

#endif /* UNIX_ERRNO2RESULT_H */
