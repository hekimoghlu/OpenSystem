/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#ifndef BSD_SYS_KDEBUG_SIGNPOST_H
#define BSD_SYS_KDEBUG_SIGNPOST_H

#include <Availability.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

#ifndef KERNEL

/*
 * kdebug_signpost(2) is deprecated.  Use the os_signpost(3) family of tracing
 * functions, instead.
 */

int kdebug_signpost(uint32_t code, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4)
__API_DEPRECATED_WITH_REPLACEMENT("os_signpost_event_emit",
    macos(10.12, 10.15), ios(10.0, 13.0), watchos(3.0, 6.0), tvos(10.0, 13.0));

int kdebug_signpost_start(uint32_t code, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4)
__API_DEPRECATED_WITH_REPLACEMENT("os_signpost_interval_begin",
    macos(10.12, 10.15), ios(10.0, 13.0), watchos(3.0, 6.0), tvos(10.0, 13.0));

int kdebug_signpost_end(uint32_t code, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4)
__API_DEPRECATED_WITH_REPLACEMENT("os_signpost_interval_end",
    macos(10.12, 10.15), ios(10.0, 13.0), watchos(3.0, 6.0), tvos(10.0, 13.0));

#endif /* !KERNEL */

__END_DECLS

#endif /* !BSD_SYS_KDEBUG_SIGNPOST_H */
