/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#include <os/system_event_log.h>
#include <sys/systm.h>
#include <sys/sysproto.h>
#include <IOKit/IOBSD.h>

int
sys_record_system_event(__unused struct proc *p, struct record_system_event_args *uap, __unused int *retval)
{
	int error = 0;

	boolean_t entitled = FALSE;
	entitled = IOCurrentTaskHasEntitlement(SYSTEM_EVENT_ENTITLEMENT);
	if (!entitled) {
		error = EPERM;
		goto done;
	}

	char event[SYSTEM_EVENT_EVENT_MAX] = {0};
	char payload[SYSTEM_EVENT_PAYLOAD_MAX] = {0};
	size_t bytes_copied;

	error = copyinstr(uap->event, event, sizeof(event), &bytes_copied);
	if (error) {
		goto done;
	}
	error = copyinstr(uap->payload, payload, sizeof(payload), &bytes_copied);
	if (error) {
		goto done;
	}

	record_system_event_no_varargs((uint8_t)(uap->type), (uint8_t)(uap->subsystem), event, payload);

done:
	return error;
}
