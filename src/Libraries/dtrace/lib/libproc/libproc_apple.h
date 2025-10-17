/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
 * Copyright 2007 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _LIBPROC_APPLE_H
#define _LIBPROC_APPLE_H

#ifdef	__cplusplus
extern "C" {
#endif
		
/*
 * APPLE NOTE: 
 *
 * This file exists to expose the innards of ps_prochandle.
 * We cannot place this in libproc.h, because it refers to
 * CoreSymbolication and mach specific classes and types.
 *
 * The Apple emulation of /proc control requires access to
 * this structure.
 */

struct ps_proc_activity_event {
	rd_event_msg_t rd_event;
	struct ps_proc_activity_event* next;
	bool synchronous;
	volatile bool destroyed;
	pthread_mutex_t synchronous_mutex;
	pthread_cond_t synchronous_cond;
};
	
struct ps_prochandle {
	pstatus_t status;
#if DTRACE_USE_CORESYMBOLICATION
	CSSymbolicatorRef symbolicator;
#endif /* DTRACE_USE_CORESYMBOLICATION */
	uint32_t current_symbol_owner_generation;
	rd_event_msg_t rd_event;
	struct ps_proc_activity_event* proc_activity_queue;
	uint32_t proc_activity_queue_enabled;
	pthread_mutex_t proc_activity_queue_mutex;
	pthread_cond_t proc_activity_queue_cond;
};
			
#ifdef  __cplusplus
}
#endif

#endif  /* _LIBPROC_APPLE_H */
