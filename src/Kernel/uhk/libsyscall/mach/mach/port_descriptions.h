/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#ifndef _MACH_PORT_DESCRIPTIONS_
#define _MACH_PORT_DESCRIPTIONS_

#include <sys/cdefs.h>

__BEGIN_DECLS

/*
 * Returns a string describing the host special port offset provided, or NULL if
 * the provided offset is not a host special port offset.
 */
const char *mach_host_special_port_description(int offset);

/*
 * Returns a string describing the task special port offset provided, or NULL if
 * the provided offset is not a task special port offset.
 */
const char *mach_task_special_port_description(int offset);

/*
 * Returns a string describing the thread special port offset provided, or NULL if
 * the provided offset is not a thread special port offset.
 */
const char *mach_thread_special_port_description(int offset);

/*
 * Returns the port for the given identifier of a host special port.  For
 * instance, passing "HOST_PRIV_PORT" would return 1.
 *
 * Returns -1 on error.
 */
int mach_host_special_port_for_id(const char *id);

/*
 * Returns the port for the given identifier of a task special port.
 *
 * Returns -1 on error.
 */
int mach_task_special_port_for_id(const char *id);

/*
 * Returns the port for the given identifier of a thread special port.
 *
 * Returns -1 on error.
 */
int mach_thread_special_port_for_id(const char *id);

__END_DECLS

#endif /* !defined(_MACH_PORT_DESCRIPTIONS_) */
