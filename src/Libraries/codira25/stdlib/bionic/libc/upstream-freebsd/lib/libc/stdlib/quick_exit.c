/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include <machine/atomic.h>
#include <stdlib.h>
#include <pthread.h>

/**
 * Linked list of quick exit handlers.  This is simpler than the atexit()
 * version, because it is not required to support C++ destructors or
 * DSO-specific cleanups.
 */
struct quick_exit_handler {
	struct quick_exit_handler *next;
	void (*cleanup)(void);
};

/**
 * Lock protecting the handlers list.
 */
static pthread_mutex_t atexit_mutex = PTHREAD_MUTEX_INITIALIZER;
/**
 * Stack of cleanup handlers.  These will be invoked in reverse order when 
 */
static struct quick_exit_handler *handlers;

int
at_quick_exit(void (*func)(void))
{
	struct quick_exit_handler *h;
	
	h = malloc(sizeof(*h));

	if (NULL == h)
		return (1);
	h->cleanup = func;
	pthread_mutex_lock(&atexit_mutex);
	h->next = handlers;
	__compiler_membar();
	handlers = h;
	pthread_mutex_unlock(&atexit_mutex);
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
	for (h = handlers; NULL != h; h = h->next) {
		__compiler_membar();
		h->cleanup();
	}
	_Exit(status);
}
