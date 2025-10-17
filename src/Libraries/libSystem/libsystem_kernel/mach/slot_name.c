/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
/*
 *	File:	slot_name.c
 *	Author:	Avadis Tevanian, Jr.
 *
 *	Copyright (C) 1987, Avadis Tevanian, Jr.
 *
 *	Convert machine slot values to human readable strings.
 *
 * HISTORY
 * 26-Jan-88  Mary Thompson (mrt) at Carnegie Mellon
 *	added case for CUP_SUBTYPE_RT_APC
 *
 * 28-Feb-87  Avadis Tevanian (avie) at Carnegie-Mellon University
 *	Created.
 *
 */

#include <mach/mach.h>
#include <stddef.h>

kern_return_t
msg_rpc(void)
{
	return KERN_FAILURE;
}

kern_return_t
msg_send(void)
{
	return KERN_FAILURE;
}

kern_return_t
msg_receive(void)
{
	return KERN_FAILURE;
}

mach_port_t
task_self_(void)
{
	return mach_task_self();
}

mach_port_t
host_self(void)
{
	return mach_host_self();
}
