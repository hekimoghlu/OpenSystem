/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
/* $Id$ */

#ifndef LWRES_ASSERT_P_H
#define LWRES_ASSERT_P_H 1

/*! \file */

#include <assert.h>		/* Required for assert() prototype. */

#define REQUIRE(x)		assert(x)
#define INSIST(x)		assert(x)

#define UNUSED(x)		((void)(x))
#define POST(x)			((void)(x))

#define SPACE_OK(b, s)		(LWRES_BUFFER_AVAILABLECOUNT(b) >= (s))
#define SPACE_REMAINING(b, s)	(LWRES_BUFFER_REMAINING(b) >= (s))

#endif /* LWRES_ASSERT_P_H */
