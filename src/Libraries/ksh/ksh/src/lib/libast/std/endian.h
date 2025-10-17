/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
 * linux/gnu compatibility
 */

#ifndef _ENDIAN_H
#define _ENDIAN_H

#include <bytesex.h>

#define	__LITTLE_ENDIAN	1234
#define	__BIG_ENDIAN	4321
#define	__PDP_ENDIAN	3412

#if defined (__USE_BSD) && !defined(__STRICT_ANSI__)

#ifndef LITTLE_ENDIAN
#define	LITTLE_ENDIAN	__LITTLE_ENDIAN
#endif

#ifndef BIG_ENDIAN
#define	BIG_ENDIAN	__BIG_ENDIAN
#endif

#ifndef PDP_ENDIAN
#define	PDP_ENDIAN	__PDP_ENDIAN
#endif

#undef	BYTE_ORDER
#define	BYTE_ORDER	__BYTE_ORDER

#endif

#endif
