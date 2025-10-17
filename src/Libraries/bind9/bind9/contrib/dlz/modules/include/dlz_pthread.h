/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef DLZ_PTHREAD_H
#define DLZ_PTHREAD_H 1

#ifndef PTHREADS
#define PTHREADS 1
#endif

#ifdef PTHREADS
#define dlz_mutex_t pthread_mutex_t
#define dlz_mutex_init pthread_mutex_init
#define dlz_mutex_destroy pthread_mutex_destroy
#define dlz_mutex_trylock pthread_mutex_trylock
#define dlz_mutex_unlock pthread_mutex_unlock
#else /* !PTHREADS */
#define dlz_mutex_t void
#define dlz_mutex_init(a, b) (0)
#define dlz_mutex_destroy(a) (0)
#define dlz_mutex_trylock(a) (0)
#define dlz_mutex_unlock(a) (0)
#endif

#endif /* DLZ_PTHREAD_H */
