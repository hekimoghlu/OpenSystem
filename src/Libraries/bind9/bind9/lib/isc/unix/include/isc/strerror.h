/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
/* $Id: strerror.h,v 1.10 2008/12/01 23:47:45 tbox Exp $ */

#ifndef ISC_STRERROR_H
#define ISC_STRERROR_H

/*! \file */

#include <sys/types.h>

#include <isc/lang.h>

ISC_LANG_BEGINDECLS

/*% String Error Size */
#define ISC_STRERRORSIZE 128

/*%
 * Provide a thread safe wrapper to strerror().
 *
 * Requires:
 * 	'buf' to be non NULL.
 */
void
isc__strerror(int num, char *buf, size_t bufsize);

ISC_LANG_ENDDECLS

#endif /* ISC_STRERROR_H */
