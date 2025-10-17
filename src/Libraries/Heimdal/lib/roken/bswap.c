/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#include <config.h>
#include "roken.h"

#ifndef HAVE_BSWAP32

ROKEN_LIB_FUNCTION unsigned int ROKEN_LIB_CALL
bswap32 (unsigned int val)
{
    return (val & 0xff) << 24 |
	(val & 0xff00) << 8 |
	(val & 0xff0000) >> 8 |
	(val & 0xff000000) >> 24;
}
#endif

#ifndef HAVE_BSWAP16

ROKEN_LIB_FUNCTION unsigned short ROKEN_LIB_CALL
bswap16 (unsigned short val)
{
    return (val & 0xff) << 8 |
	(val & 0xff00) >> 8;
}
#endif
