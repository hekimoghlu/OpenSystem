/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
#include <sys/cdefs.h>

typedef enum {
	KDP_EVENT_ENTER,
	KDP_EVENT_EXIT,
	KDP_EVENT_PANICLOG
} kdp_event_t;

__BEGIN_DECLS
typedef void (*kdp_callout_fn_t)(void *arg, kdp_event_t event);

/*
 * Register fn(arg, event) to be called at kdp entry/exit.
 * The callouts are made in a single-threaded environment, interrupts are
 * disabled and processors other than the callout processor quiesced.
 * N.B. callouts are strictly limited in what they can do: they must execute
 * with interrupts disabled and they can't call back into the kernel for any
 * non-trivial service.
 */
extern void kdp_register_callout(kdp_callout_fn_t fn, void *arg);

__END_DECLS
