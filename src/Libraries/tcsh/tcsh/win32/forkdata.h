/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#ifndef FORK_DATA_H
#define FORK_DATA_H

#include <setjmp.h>

/* 
 * This structure is copied by fork() to the child process. It 
 * contains variables of national importance
 *
 * Thanks to Mark Tucker for the idea. tcsh now finally works on
 * alphas.
 * -amol
 */
typedef struct _fork_data {
	unsigned long _forked;
	void  *_fork_stack_begin;
	void  *_fork_stack_end;
	unsigned long _heap_size;
	HANDLE _hforkparent, _hforkchild;
	void * _heap_base;
	void * _heap_top;
	jmp_buf _fork_context;
} ForkData;

#define __forked gForkData._forked
#define __fork_stack_begin gForkData._fork_stack_begin
#define __fork_stack_end gForkData._fork_stack_end
#define __hforkparent gForkData._hforkparent
#define __hforkchild gForkData._hforkchild
#define __fork_context gForkData._fork_context
#define __heap_base gForkData._heap_base
#define __heap_size gForkData._heap_size
#define __heap_top gForkData._heap_top

extern ForkData gForkData;

#ifdef NTDBG
#define FORK_TIMEOUT INFINITE
#else
#define FORK_TIMEOUT (50000)
#endif /*!NTDBG */



#endif FORK_DATA_H
