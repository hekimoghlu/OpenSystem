/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#include "debug_counter.h"
#if USE_DEBUG_COUNTER
#include <stdio.h>
#include <locale.h>
#include "internal.h"

static const char *const debug_counter_names[] = {
    ""
#define RB_DEBUG_COUNTER(name) #name,
#include "debug_counter.h"
#undef RB_DEBUG_COUNTER
};

size_t rb_debug_counter[numberof(debug_counter_names)];

void
rb_debug_counter_show_results(const char *msg)
{
    const char *env = getenv("RUBY_DEBUG_COUNTER_DISABLE");

    setlocale(LC_NUMERIC, "");

    if (env == NULL || strcmp("1", env) != 0) {
	int i;
        fprintf(stderr, "[RUBY_DEBUG_COUNTER]\t%d %s\n", getpid(), msg);
	for (i=0; i<RB_DEBUG_COUNTER_MAX; i++) {
            fprintf(stderr, "[RUBY_DEBUG_COUNTER]\t%-30s\t%'14"PRIuSIZE"\n",
		    debug_counter_names[i],
		    rb_debug_counter[i]);
	}
    }
}

__attribute__((destructor))
static void
debug_counter_show_results_at_exit(void)
{
    rb_debug_counter_show_results("normal exit.");
}
#else
void
rb_debug_counter_show_results(const char *msg)
{
}
#endif /* USE_DEBUG_COUNTER */
