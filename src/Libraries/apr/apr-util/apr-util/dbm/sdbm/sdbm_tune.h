/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
/*
 * sdbm - ndbm work-alike hashed database library
 * tuning and portability constructs [not nearly enough]
 * author: oz@nexus.yorku.ca
 */

#ifndef SDBM_TUNE_H
#define SDBM_TUNE_H

#include "apr_errno.h"

/* ### this might be better off as sizeof(char *) */
#define BYTESIZ		8

/*
 * misc
 */
#ifdef DEBUG
#define debug(x)	printf x
#else
#define debug(x)
#endif

#endif /* SDBM_TUNE_H */
