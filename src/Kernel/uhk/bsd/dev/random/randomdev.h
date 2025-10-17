/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
 *       WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
 *
 *       THIS FILE IS NEEDED TO PASS FIPS ACCEPTANCE FOR THE RANDOM NUMBER GENERATOR.
 *       IF YOU ALTER IT IN ANY WAY, WE WILL NEED TO GO THOUGH FIPS ACCEPTANCE AGAIN,
 *       AN OPERATION THAT IS VERY EXPENSIVE AND TIME CONSUMING.  IN OTHER WORDS,
 *       DON'T MESS WITH THIS FILE.
 *
 *       WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
 */

#ifndef __DEV_RANDOMDEV_H__
#define __DEV_RANDOMDEV_H__

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_PRIVATE

#include <sys/random.h>

void PreliminarySetup( void );
void random_init( void );
int random_open(dev_t dev, int flags, int devtype, struct proc *pp);
int random_close(dev_t dev, int flags, int mode, struct proc *pp);
int random_read(dev_t dev, struct uio *uio, int ioflag);
int random_write(dev_t dev, struct uio *uio, int ioflag);

u_int32_t RandomULong( void );

#endif /* __APPLE_API_PRIVATE */
#endif /* __DEV_RANDOMDEV_H__ */
