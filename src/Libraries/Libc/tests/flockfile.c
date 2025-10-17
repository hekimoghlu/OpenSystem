/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "darwintest.h"

T_DECL(flockfile_preserve_errno, "flockfile preserves errno")
{
	errno = EBUSY;
	flockfile(stderr);
	T_ASSERT_EQ(errno, EBUSY, "flockfile preserves errno");
}

T_DECL(funlockfile_preserve_errno, "funlockfile preserves errno")
{
	errno = EBUSY;
	funlockfile(stderr);
	T_ASSERT_EQ(errno, EBUSY, "funlockfile preserves errno");
}

T_DECL(ftrylockfile_preserve_errno, "ftrylockfile preserves errno")
{
	errno = EBUSY;
	ftrylockfile(stderr);
	T_ASSERT_EQ(errno, EBUSY, "ftrylockfile preserves errno");
}

