/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include <stdlib.h>
#include <Block.h>
#include <dispatch/dispatch.h>

#include "sec_action.h"

struct sec_action_context_s {
	dispatch_queue_t queue;
	dispatch_source_t source;
	dispatch_block_t handler;
	uint64_t interval;
};

static void _sec_action_event(void *ctx);
static void _sec_action_finalize(void *ctx);

#pragma mark -

sec_action_t
sec_action_create_with_queue(dispatch_queue_t queue, const char *label, uint64_t interval)
{
	dispatch_source_t source;
	struct sec_action_context_s *context;
	if (queue) {
		dispatch_retain(queue);
	} else {
		queue = dispatch_queue_create(label, DISPATCH_QUEUE_SERIAL);
	}
	source = dispatch_source_create(DISPATCH_SOURCE_TYPE_DATA_ADD, 0, 0, queue);

	context = malloc(sizeof(*context));
	context->queue = queue;
	context->source = source;
	context->handler = NULL;
	context->interval = interval;
	dispatch_set_context(source, context);

	dispatch_source_set_event_handler_f(source, _sec_action_event);
	dispatch_set_finalizer_f(source, _sec_action_finalize);

	return source;
}

sec_action_t
sec_action_create(const char *label, uint64_t interval)
{
	return sec_action_create_with_queue(NULL, label, interval);
}

void
sec_action_set_handler(sec_action_t source, dispatch_block_t handler)
{
	struct sec_action_context_s *context;

	context = dispatch_get_context(source);
	context->handler = Block_copy(handler);

	dispatch_activate(source);
}

void
sec_action_perform(sec_action_t source)
{
	dispatch_source_merge_data(source, 1);
}

#pragma mark -

static void
resume(void *ctx)
{
    dispatch_resume(ctx);
}

static void
_sec_action_event(void *ctx)
{
	struct sec_action_context_s *context = (struct sec_action_context_s *)ctx;

	if (context->handler != NULL) {
		context->handler();
	}

	// Suspend the source; resume after specified interval.
	dispatch_suspend(context->source);
	dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(context->interval * NSEC_PER_SEC)), context->queue, context->source, resume);
}

static void
_sec_action_finalize(void *ctx)
{
	struct sec_action_context_s *context = (struct sec_action_context_s *)ctx;

	dispatch_release(context->queue);
	if (context->handler != NULL) {
		Block_release(context->handler);
	}
	free(context);
}
