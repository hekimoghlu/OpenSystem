/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include <sys/types.h>

#include <stdatomic.h>
#include <stdlib.h>

/**
 * Linked list of quick exit handlers.  These will be invoked in reverse
 * order of insertion when quick_exit() is called.  This is simpler than
 * the atexit() version, because it is not required to support C++
 * destructors or DSO-specific cleanups.
 */
struct quick_exit_handler {
	struct quick_exit_handler *next;
	void (*cleanup)(void);
};

static _Atomic(struct quick_exit_handler *) handlers;

int
at_quick_exit(void (*func)(void))
{
	struct quick_exit_handler *h;

	if ((h = calloc(1, sizeof(*h))) == NULL) {
		return (-1);
	}
	h->cleanup = func;
	while (!atomic_compare_exchange_strong(&handlers, &h->next, h)) {
		/* nothing */ ;
	}
	return (0);
}

void
quick_exit(int status)
{
	struct quick_exit_handler *h;

	/*
	 * XXX: The C++ spec requires us to call std::terminate if there is an
	 * exception here.
	 */
	for (h = atomic_load_explicit(&handlers, memory_order_acquire);
	     h != NULL; h = h->next) {
		h->cleanup();
	}
	_Exit(status);
}
