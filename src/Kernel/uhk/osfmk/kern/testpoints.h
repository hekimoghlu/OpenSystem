/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#ifndef _TESTPOINTS_H_
#define _TESTPOINTS_H_

#ifdef KERNEL
#include <kern/kern_types.h>
#include <kern/assert.h>
#include <kern/locks.h>
#else /* KERNEL */
#include <sys/types.h>
#include <assert.h>
#endif /* KERNEL */

#include <sys/cdefs.h>

/* Testpoints are intended as a generic mechanism which would allow developers
 * to call a test code from various places of source code by inserting one line
 * only. What will happen when testpoint is called is determined by current
 * test scenario which can be set via sysctl. */

/* Testpoints scenarios */
__enum_decl(tps_id_t, uint64_t, {
	TPS_NONE,                       // nothing enabled
	TPS_STACKSHOT_UPCALL,           // c-hello-exclaves thread is going to hang in upcall during stackshot and tries to return as soon as it can
	TPS_STACKSHOT_LONG_UPCALL,      // c-hello-exclaves thread is going to hang in upcall during stackshot and returns when stackshot is completely done
});

/* Testpoint definitions */
__enum_decl(tp_id_t, uint16_t, {
	TP_BLOCK_START_STACKSHOT,       // this action will mark stackshot blocked
	TP_WAIT_START_STACKSHOT,        // stackshot thread is waiting here until test thread is in upcall
	TP_UPCALL,                      // upcall handler, unblocks stackshot and waits
	TP_START_COLLECTION,            // just before exclave threads collection starts, unblocks upcall and waits
	TP_AST,                         // before going back to exclaves, unblocks exclaves collection
	TP_STACKSHOT_DONE,              // stackshot is done
	TESTPOINT_COUNT,
});

typedef uint32_t tp_val_t;

// must fit 64 bits
typedef struct tp_sysctl_msg {
	uint16_t id;
	uint16_t _unused;
	uint32_t val;
} tp_sysctl_msg_t;

static_assert(sizeof(uint64_t) == sizeof(tp_sysctl_msg_t), "tp_sysctl_msg_t does have 64 bits");


#if DEBUG || DEVELOPMENT || TESTPOINTS

/* Action is determined by current scenario */
void
tp_call(tp_id_t testpoint, tp_val_t val);

#define TESTPOINT(x) tp_call(x, 0);

#else /* DEBUG || DEVELOPMENT || TESTPOINTS */

#define TESTPOINT(x) ;

#endif /* DEBUG || DEVELOPMENT || TESTPOINTS */

#if (DEBUG || DEVELOPMENT) && KERNEL

/* Below functions are intended for kernel tests implementations. They are not
 * thread safe and should be protected by tp_mtx. See exclaves_test_stackshot.c
 * for sample code. */

extern lck_mtx_t tp_mtx;

/* Set given testpoint value to 1. */
void
tp_block(tp_id_t testpoint);

/* Set given testpoint value to 0 and wakeup waiting threads. */
void
tp_unblock(tp_id_t other_testpoint);

/* Wait until given testpoint value is zero. */
void
tp_wait(tp_id_t testpoint);

/* Unblock other testpoint, then block and wait. */
void
tp_relay(tp_id_t testpoint, tp_id_t other_testpoint);

#endif /* (DEBUG || DEVELOPMENT) && KERNEL */

#endif /* _TESTPOINTS_H_ */
