/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef MEMSTATS_H
#define MEMSTATS_H

#include "statistic.h"

#ifndef TOP_ANONYMOUS_MEMORY
struct statistic *top_rsize_create(WINDOW *parent, const char *name);
#endif /* !TOP_ANONYMOUS_MEMORY */
struct statistic *top_vsize_create(WINDOW *parent, const char *name);
struct statistic *top_rprvt_create(WINDOW *parent, const char *name);
struct statistic *top_vprvt_create(WINDOW *parent, const char *name);
#ifndef TOP_ANONYMOUS_MEMORY
struct statistic *top_rshrd_create(WINDOW *parent, const char *name);
#endif /* !TOP_ANONYMOUS_MEMORY */
struct statistic *top_mregion_create(WINDOW *parent, const char *name);
struct statistic *top_pageins_create(WINDOW *parent, const char *name);
struct statistic *top_kprvt_create(WINDOW *parent, const char *name);
struct statistic *top_kshrd_create(WINDOW *parent, const char *name);
#ifdef TOP_ANONYMOUS_MEMORY
struct statistic *top_pmem_create(WINDOW *parent, const char *name);
struct statistic *top_purg_create(WINDOW *parent, const char *name);
struct statistic *top_compressed_create(WINDOW *parent, const char *name);
#endif /* TOP_ANONYMOUS_MEMORY */

struct statistic *top_jetsam_priority_create(WINDOW *parent, const char *name);

#endif /*MEMSTATS_H*/
