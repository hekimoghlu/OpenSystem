/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#ifndef _memchanDecls_h_INCLUDE
#define _memchanDecls_h_INCLUDE

/* !BEGIN!: Do not edit below this line. */

/*
 * Exported function declarations:
 */

/* 0 */
EXTERN int		Memchan_Init _ANSI_ARGS_((Tcl_Interp * interp));
/* 1 */
EXTERN int		Memchan_SafeInit _ANSI_ARGS_((Tcl_Interp * interp));
/* 2 */
EXTERN Tcl_Channel	Memchan_CreateMemoryChannel _ANSI_ARGS_((
				Tcl_Interp * interp, int initialSize));
/* 3 */
EXTERN Tcl_Channel	Memchan_CreateFifoChannel _ANSI_ARGS_((
				Tcl_Interp * interp));
/* 4 */
EXTERN void		Memchan_CreateFifo2Channel _ANSI_ARGS_((
				Tcl_Interp * interp, Tcl_Channel * aPtr, 
				Tcl_Channel * bPtr));
/* 5 */
EXTERN Tcl_Channel	Memchan_CreateZeroChannel _ANSI_ARGS_((
				Tcl_Interp * interp));
/* 6 */
EXTERN Tcl_Channel	Memchan_CreateNullChannel _ANSI_ARGS_((
				Tcl_Interp * interp));
/* 7 */
EXTERN Tcl_Channel	Memchan_CreateRandomChannel _ANSI_ARGS_((
				Tcl_Interp * interp));

typedef struct MemchanStubs {
    int magic;
    struct MemchanStubHooks *hooks;

    int (*memchan_Init) _ANSI_ARGS_((Tcl_Interp * interp)); /* 0 */
    int (*memchan_SafeInit) _ANSI_ARGS_((Tcl_Interp * interp)); /* 1 */
    Tcl_Channel (*memchan_CreateMemoryChannel) _ANSI_ARGS_((Tcl_Interp * interp, int initialSize)); /* 2 */
    Tcl_Channel (*memchan_CreateFifoChannel) _ANSI_ARGS_((Tcl_Interp * interp)); /* 3 */
    void (*memchan_CreateFifo2Channel) _ANSI_ARGS_((Tcl_Interp * interp, Tcl_Channel * aPtr, Tcl_Channel * bPtr)); /* 4 */
    Tcl_Channel (*memchan_CreateZeroChannel) _ANSI_ARGS_((Tcl_Interp * interp)); /* 5 */
    Tcl_Channel (*memchan_CreateNullChannel) _ANSI_ARGS_((Tcl_Interp * interp)); /* 6 */
    Tcl_Channel (*memchan_CreateRandomChannel) _ANSI_ARGS_((Tcl_Interp * interp)); /* 7 */
} MemchanStubs;

#ifdef __cplusplus
extern "C" {
#endif
extern MemchanStubs *memchanStubsPtr;
#ifdef __cplusplus
}
#endif

#if defined(USE_MEMCHAN_STUBS) && !defined(USE_MEMCHAN_STUB_PROCS)

/*
 * Inline function declarations:
 */

#ifndef Memchan_Init
#define Memchan_Init \
	(memchanStubsPtr->memchan_Init) /* 0 */
#endif
#ifndef Memchan_SafeInit
#define Memchan_SafeInit \
	(memchanStubsPtr->memchan_SafeInit) /* 1 */
#endif
#ifndef Memchan_CreateMemoryChannel
#define Memchan_CreateMemoryChannel \
	(memchanStubsPtr->memchan_CreateMemoryChannel) /* 2 */
#endif
#ifndef Memchan_CreateFifoChannel
#define Memchan_CreateFifoChannel \
	(memchanStubsPtr->memchan_CreateFifoChannel) /* 3 */
#endif
#ifndef Memchan_CreateFifo2Channel
#define Memchan_CreateFifo2Channel \
	(memchanStubsPtr->memchan_CreateFifo2Channel) /* 4 */
#endif
#ifndef Memchan_CreateZeroChannel
#define Memchan_CreateZeroChannel \
	(memchanStubsPtr->memchan_CreateZeroChannel) /* 5 */
#endif
#ifndef Memchan_CreateNullChannel
#define Memchan_CreateNullChannel \
	(memchanStubsPtr->memchan_CreateNullChannel) /* 6 */
#endif
#ifndef Memchan_CreateRandomChannel
#define Memchan_CreateRandomChannel \
	(memchanStubsPtr->memchan_CreateRandomChannel) /* 7 */
#endif

#endif /* defined(USE_MEMCHAN_STUBS) && !defined(USE_MEMCHAN_STUB_PROCS) */

/* !END!: Do not edit above this line. */

#endif /* _memchanDecls_h_INCLUDE */
