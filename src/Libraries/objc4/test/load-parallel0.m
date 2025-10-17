/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

#ifndef N
#error -DN=n missing
#endif

#import <objc/objc-api.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>
#include "test.h"
extern atomic_int state;

#define CLASS0(n,nn)                                                    \
    OBJC_ROOT_CLASS                                                     \
    @interface C_##n##_##nn @end                                        \
    @implementation C_##n##_##nn                                        \
    +(void)load {                                                       \
        atomic_fetch_add_explicit(&state, 1, memory_order_relaxed);     \
        usleep(10); }                                                   \
    @end

#define CLASS(n,nn) CLASS0(n,nn)

CLASS(a,N)
CLASS(b,N)
CLASS(c,N)
CLASS(d,N)
CLASS(e,N)
CLASS(f,N)
CLASS(g,N)
CLASS(h,N)
CLASS(i,N)
CLASS(j,N)
CLASS(k,N)
CLASS(l,N)
CLASS(m,N)
CLASS(n,N)
CLASS(o,N)
CLASS(p,N)
CLASS(q,N)
CLASS(r,N)
CLASS(s,N)
CLASS(t,N)
CLASS(u,N)
CLASS(v,N)
CLASS(w,N)
CLASS(x,N)
CLASS(y,N)
CLASS(z,N)
