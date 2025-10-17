/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#include <pexpert/pexpert.h>
#include <pexpert/protos.h>
#include <machine/machine_routines.h>

#if CONFIG_DTRACE && DEVELOPMENT
#include <mach/sdt.h>
#endif

void PE_incoming_interrupt(int);


struct i386_interrupt_handler {
	IOInterruptHandler      handler;
	void                    *nub;
	void                    *target;
	void                    *refCon;
};

typedef struct i386_interrupt_handler i386_interrupt_handler_t;

i386_interrupt_handler_t        PE_interrupt_handler;



void
PE_incoming_interrupt(int interrupt)
{
	i386_interrupt_handler_t        *vector;

	vector = &PE_interrupt_handler;

#if CONFIG_DTRACE && DEVELOPMENT
	DTRACE_INT5(interrupt_start, void *, vector->nub, int, 0,
	    void *, vector->target, IOInterruptHandler, vector->handler,
	    void *, vector->refCon);
#endif

	vector->handler(vector->target, NULL, vector->nub, interrupt);

#if CONFIG_DTRACE && DEVELOPMENT
	DTRACE_INT5(interrupt_complete, void *, vector->nub, int, 0,
	    void *, vector->target, IOInterruptHandler, vector->handler,
	    void *, vector->refCon);
#endif
}

void
PE_install_interrupt_handler(void *nub,
    __unused int source,
    void *target,
    IOInterruptHandler handler,
    void *refCon)
{
	i386_interrupt_handler_t        *vector;

	vector = &PE_interrupt_handler;

	/*vector->source = source; IGNORED */
	vector->handler = handler;
	vector->nub = nub;
	vector->target = target;
	vector->refCon = refCon;
}
