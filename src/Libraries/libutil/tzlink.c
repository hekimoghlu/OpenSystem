/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#include <TargetConditionals.h>
#include <xpc/xpc.h>
#include <errno.h>

#include "tzlink.h"
#include "tzlink_internal.h"

errno_t
tzlink(const char *tz)
{
#if TARGET_OS_IPHONE || TARGET_OS_OSX && !TARGET_OS_SIMULATOR
	xpc_connection_t connection;
	xpc_object_t request, reply;
	errno_t e;

	if (tz == NULL) {
		return EINVAL;
	}

	connection = xpc_connection_create_mach_service(TZLINK_SERVICE_NAME, NULL, XPC_CONNECTION_MACH_SERVICE_PRIVILEGED);
	xpc_connection_set_event_handler(connection, ^(__unused xpc_object_t event) {
	});
	xpc_connection_resume(connection);

	request = xpc_dictionary_create(NULL, NULL, 0);
	xpc_dictionary_set_string(request, TZLINK_KEY_REQUEST_TIMEZONE, tz);

	reply = xpc_connection_send_message_with_reply_sync(connection, request);
	if (xpc_get_type(reply) == XPC_TYPE_DICTIONARY) {
		e = (errno_t)xpc_dictionary_get_uint64(reply, TZLINK_KEY_REPLY_ERROR);
	} else {
		e = EIO;
	}

	xpc_release(reply);
	xpc_release(request);
	xpc_release(connection);

	return e;
#else /* !TARGET_OS_IPHONE */
#pragma unused (tz)
	return ENOTSUP;
#endif /* TARGET_OS_IPHONE */
}
