/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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

// Copyright (c) 2020 Apple Computer, Inc. All rights reserved.

#include <stdlib.h>
#include <os/atomic_private.h>

#define kheap_alloc(h, s, f) calloc(1, s)
#define kfree(p, s) free(p)
#define kalloc_type(t, f) calloc(1, sizeof(t))
#define kfree_type(t, p) free(p)
#define kalloc_data(s, f) calloc(1, s)
#define kfree_data(p, s) free(p)
#define panic(...) T_ASSERT_FAIL(__VA_ARGS__)
#define PE_i_can_has_debugger(...) true
#define SECURITY_READ_ONLY_LATE(X) X
#define __startup_func

#define ml_get_cpu_count() 6
