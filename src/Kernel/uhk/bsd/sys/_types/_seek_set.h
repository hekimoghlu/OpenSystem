/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef _SEEK_SET_H_
#define _SEEK_SET_H_

#include <sys/cdefs.h>

/* whence values for lseek(2) */
#ifndef SEEK_SET
#define SEEK_SET        0       /* set file offset to offset */
#define SEEK_CUR        1       /* set file offset to current plus offset */
#define SEEK_END        2       /* set file offset to EOF plus offset */
#endif  /* !SEEK_SET */

#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL
#ifndef SEEK_HOLE
#define SEEK_HOLE       3       /* set file offset to the start of the next hole greater than or equal to the supplied offset */
#endif

#ifndef SEEK_DATA
#define SEEK_DATA       4       /* set file offset to the start of the next non-hole file region greater than or equal to the supplied offset */
#endif
#endif /* __DARWIN_C_LEVEL >= __DARWIN_C_FULL */

#endif /* _SEEK_SET_H_ */
