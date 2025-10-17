/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#ifndef SUDO_RAND_H
#define SUDO_RAND_H

#include <stdlib.h>	/* For arc4random() on systems that have it */

/*
 * All libc replacements are prefixed with "sudo_" to avoid namespace issues.
 */

#ifndef HAVE_ARC4RANDOM
sudo_dso_public uint32_t sudo_arc4random(void);
# undef arc4random
# define arc4random() sudo_arc4random()
#endif /* ARC4RANDOM */

#ifndef HAVE_ARC4RANDOM_BUF
sudo_dso_public void sudo_arc4random_buf(void *buf, size_t n);
# undef arc4random_buf
# define arc4random_buf(a, b) sudo_arc4random_buf((a), (b))
#endif /* ARC4RANDOM_BUF */

#ifndef HAVE_ARC4RANDOM_UNIFORM
sudo_dso_public uint32_t sudo_arc4random_uniform(uint32_t upper_bound);
# undef arc4random_uniform
# define arc4random_uniform(_a) sudo_arc4random_uniform((_a))
#endif /* ARC4RANDOM_UNIFORM */

#ifndef HAVE_GETENTROPY
/* Note: not exported by libutil. */
int sudo_getentropy(void *buf, size_t buflen);
# undef getentropy
# define getentropy(_a, _b) sudo_getentropy((_a), (_b))
#endif /* HAVE_GETENTROPY */

#endif /* SUDO_RAND_H */
