/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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

//
//  parallel_register_cancel.c
//  Libnotify
//

#include <darwintest.h>
#include <dispatch/dispatch.h>
#include <notify.h>
#include <stdio.h>

T_DECL(parallel_register_cancel,
       "parallel register/cancel test",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"))
{
	dispatch_queue_t noteQueue = dispatch_queue_create("noteQ", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
	static int tokens[100000];
	dispatch_apply(100000, DISPATCH_APPLY_AUTO, ^(size_t i) {
        assert(notify_register_check("com.example.test", &tokens[i]) == NOTIFY_STATUS_OK);
        assert(notify_cancel(tokens[i]) == NOTIFY_STATUS_OK);
		assert(notify_register_dispatch("com.example.test", &tokens[i], noteQueue, ^(int i){}) == NOTIFY_STATUS_OK);
		assert(notify_cancel(tokens[i]) == NOTIFY_STATUS_OK);
		assert(notify_post("com.example.test") == NOTIFY_STATUS_OK);
	});


	dispatch_apply(100000, DISPATCH_APPLY_AUTO, ^(size_t i) {
		assert(notify_register_check("self.example.test", &tokens[i]) == NOTIFY_STATUS_OK);
		assert(notify_cancel(tokens[i]) == NOTIFY_STATUS_OK);
		assert(notify_register_dispatch("self.example.test", &tokens[i], noteQueue, ^(int i){}) == NOTIFY_STATUS_OK);
		assert(notify_cancel(tokens[i]) == NOTIFY_STATUS_OK);
		assert(notify_post("self.example.test") == NOTIFY_STATUS_OK);
	});

	dispatch_release(noteQueue);

	T_PASS("Success");
}
