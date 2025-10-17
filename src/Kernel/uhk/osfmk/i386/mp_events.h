/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef __AT386_MP_EVENTS__
#define __AT386_MP_EVENTS__

/* Interrupt types */

#ifndef ASSEMBLER

#include <sys/cdefs.h>

typedef enum {
	MP_TLB_FLUSH = 0,
	MP_KDP,
	MP_KDB,
	MP_AST,
	MP_IDLE,
	MP_UNIDLE,
	MP_CALL,
	MP_CALL_PM,
	MP_LAST
} mp_event_t;

#define MP_EVENT_NAME_DECL()    \
const char *mp_event_name[] = { \
	"MP_TLB_FLUSH",         \
	"MP_KDP",               \
	"MP_KDB",               \
	"MP_AST",               \
	"MP_IDLE",              \
	"MP_UNIDLE",            \
	"MP_CALL",              \
	"MP_CALL_PM",           \
	"MP_LAST"               \
}

typedef enum { SYNC, ASYNC, NOSYNC } mp_sync_t;

__BEGIN_DECLS

extern void     i386_signal_cpu(int cpu, mp_event_t event, mp_sync_t mode);
extern void     i386_activate_cpu(void);
extern void     i386_deactivate_cpu(void);
extern void     cpu_NMI_interrupt(int /* cpu */);

__END_DECLS

#endif

#endif
