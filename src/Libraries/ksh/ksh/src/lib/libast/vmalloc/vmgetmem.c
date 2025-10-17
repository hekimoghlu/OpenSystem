/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include	<vmalloc.h>

/*
 * vm open/close/resize - a handy default for discipline memory functions
 *
 *	vmgetmem(0,0,0)		open new region
 *	vmgetmem(r,0,0)		free region
 *	vmgetmem(r,0,n)		allocate n bytes initialized to 0
 *	vmgetmem(r,p,0)		free p
 *	vmgetmem(r,p,n)		realloc p to n bytes
 *
 * Written by Glenn S. Fowler.
 */

#if __STD_C
Void_t* vmgetmem(Vmalloc_t* vm, Void_t* data, size_t size)
#else
Void_t* vmgetmem(vm, data, size)
Vmalloc_t*	vm;
Void_t*		data;
size_t		size;
#endif
{
	if (!vm)
		return vmopen(Vmdcheap, Vmbest, 0);
	if (data || size)
		return vmresize(vm, data, size, VM_RSMOVE|VM_RSCOPY|VM_RSZERO);
	vmclose(vm);
	return 0;
}
