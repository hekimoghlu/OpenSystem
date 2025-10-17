/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef _FILE_H_
#define _FILE_H_

#include <fcntl.h>

#ifndef L_SET
# define L_SET  0       /* seek from beginning.  */
# define L_CURR	1       /* seek from current position.  */
# define L_INCR	1       /* ditto.  */
# define L_XTND 2       /* seek from end.  */
#endif

#ifndef R_OK
# define R_OK  4        /* test whether readable.  */
# define W_OK  2        /* test whether writable.  */
# define X_OK  1        /* test whether executable. */
# define F_OK  0        /* test whether exist.  */
#endif

#endif
