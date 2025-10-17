/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef __os_system_event_log_h
#define __os_system_event_log_h

#include <Availability.h>
#include <os/object.h>
#include <stdint.h>
#include <sys/event_log.h>

__BEGIN_DECLS

/* uncrustify weirdly indents the first availability line */
/* BEGIN IGNORE CODESTYLE */
__WATCHOS_AVAILABLE(9.0) __OSX_AVAILABLE(13.0) __IOS_AVAILABLE(16.0) __TVOS_AVAILABLE(16.0)
/* END IGNORE CODESTYLE */
OS_EXPORT OS_NOTHROW
void
record_system_event(uint8_t type, uint8_t subsystem, const char *event, const char *format, ...) __printflike(4, 5);

__WATCHOS_AVAILABLE(9.0) __OSX_AVAILABLE(13.0) __IOS_AVAILABLE(16.0) __TVOS_AVAILABLE(16.0)
OS_EXPORT OS_NOTHROW
void
record_system_event_no_varargs(uint8_t type, uint8_t subsystem, const char *event, const char *payload);

__END_DECLS

#endif /* __os_system_event_log_h */
