/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#ifndef _SYSTEM_EVENT_LOG_H_
#define _SYSTEM_EVENT_LOG_H_

#include <Availability.h>
#include <sys/cdefs.h>
#include <stdint.h>

__BEGIN_DECLS

#define SYSTEM_EVENT_ENTITLEMENT "com.apple.private.record_system_event"

// These arbitrary numbers are used to limit the size of the system event message.
// We don't want the messages to be too long, as they're going over serial.
#define SYSTEM_EVENT_EVENT_MAX 64
#define SYSTEM_EVENT_PAYLOAD_MAX 96

__enum_decl(system_event_type, uint8_t, {
	SYSTEM_EVENT_TYPE_FIRST = 0,
	SYSTEM_EVENT_TYPE_INFO,
	SYSTEM_EVENT_TYPE_ERROR,
	SYSTEM_EVENT_TYPE_LAST
});

__enum_decl(system_event_subsystem, uint8_t, {
	SYSTEM_EVENT_SUBSYSTEM_FIRST = 0,
	SYSTEM_EVENT_SUBSYSTEM_LAUNCHD,
	SYSTEM_EVENT_SUBSYSTEM_TEST,
	SYSTEM_EVENT_SUBSYSTEM_NVRAM,
	SYSTEM_EVENT_SUBSYSTEM_PROCESS,
	SYSTEM_EVENT_SUBSYSTEM_PMRD,
	SYSTEM_EVENT_SUBSYSTEM_LAST
});

#ifndef KERNEL

/*
 * Known subsystems can use this to emit system event transition logging messages in
 * a structured and understandable way.
 */

__WATCHOS_AVAILABLE(9.0) __OSX_AVAILABLE(13.0) __IOS_AVAILABLE(16.0) __TVOS_AVAILABLE(16.0)
int
record_system_event_as_kernel(system_event_type type, system_event_subsystem subsystem, const char *event,
    const char *payload);

#endif /* !KERNEL */

__END_DECLS

#endif /* _SYSTEM_EVENT_LOG_H_ */
