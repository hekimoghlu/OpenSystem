/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
 * standalone mini vmalloc interface
 */

#ifndef _VMALLOC_H
#define _VMALLOC_H		1

#define vmalloc(v,n)		_vm_resize(v,(void*)0,n)
#define vmalign(v,n,a)		_vm_resize(v,(void*)0,n)
#define vmclose(v)		_vm_close(v)
#define vmfree(v,p)
#define vmnewof(v,o,t,n,x)	(t*)_vm_resize(v,(void*)o,sizeof(t)*(n)+(x))
#define vmopen(a,b,c)		_vm_open()

#define VM_CHUNK		(32*1024)
#define VM_ALIGN		16

typedef struct Vmchunk_s
{
	struct Vmchunk_s*	next;
	char			align[VM_ALIGN - sizeof(struct Vmchunk_s*)];
	char			data[VM_CHUNK - VM_ALIGN];
} Vmchunk_t;

typedef struct Vmalloc_s
{
	Vmchunk_t		base;		
	Vmchunk_t*		current;
	char*			data;
	long			size;
	long			last;
} Vmalloc_t;

extern Vmalloc_t*		Vmregion;

extern int			_vm_close(Vmalloc_t*);
extern Vmalloc_t*		_vm_open(void);
extern void*			_vm_resize(Vmalloc_t*, void*, unsigned long);

#endif
