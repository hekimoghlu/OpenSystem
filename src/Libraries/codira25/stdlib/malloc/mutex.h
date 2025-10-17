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

#ifndef MUTEX_H
#define MUTEX_H

#include <pthread.h>

#include "util.h"

struct mutex {
    pthread_mutex_t lock;
};

#define MUTEX_INITIALIZER (struct mutex){PTHREAD_MUTEX_INITIALIZER}

static inline void mutex_init(struct mutex *m) {
    if (unlikely(pthread_mutex_init(&m->lock, NULL))) {
        fatal_error("mutex initialization failed");
    }
}

static inline void mutex_lock(struct mutex *m) {
    pthread_mutex_lock(&m->lock);
}

static inline void mutex_unlock(struct mutex *m) {
    pthread_mutex_unlock(&m->lock);
}

#endif
