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
#include "IPConfigurationAgentUtil.h"
#include "symbol_scope.h"

STATIC CFRunLoopRef	S_IPConfigurationAgentRunLoop;

PRIVATE_EXTERN CFRunLoopRef
IPConfigurationAgentRunLoop(void)
{
	return (S_IPConfigurationAgentRunLoop);
}

PRIVATE_EXTERN void
IPConfigurationAgentSetRunLoop(CFRunLoopRef runloop)
{
	S_IPConfigurationAgentRunLoop = runloop;
}

PRIVATE_EXTERN dispatch_queue_t
IPConfigurationAgentQueue(void)
{
	STATIC dispatch_queue_t	queue;

	if (queue == NULL) {
		queue = dispatch_queue_create("IPConfigurationAgentQueue",
					      NULL);
	}
	return (queue);
}
