/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
/* { dg-do run { xfail mips*-*-* arm*-*-* strongarm*-*-* xscale*-*-* } } */
#include "ffitest.h"

void* cls_pointer_fn(void* a1, void* a2)
{
	void*	result	= (void*)((long)a1 + (long)a2);

	printf("0x%08x 0x%08x: 0x%08x\n", a1, a2, result);

	return result;
}

static void
cls_pointer_gn(ffi_cif* cif, void* resp, void** args, void* userdata)
{
	void*	a1	= *(void**)(args[0]);
	void*	a2	= *(void**)(args[1]);

	*(void**)resp = cls_pointer_fn(a1, a2);
}

int main (void)
{
	ffi_cif	cif;
#ifndef USING_MMAP
	static ffi_closure	cl;
#endif
	ffi_closure*	pcl;
	void*			args[3];
//	ffi_type		cls_pointer_type;
	ffi_type*		arg_types[3];

#ifdef USING_MMAP
	pcl = allocate_mmap(sizeof(ffi_closure));
#else
	pcl = &cl;
#endif

/*	cls_pointer_type.size = sizeof(void*);
	cls_pointer_type.alignment = 0;
	cls_pointer_type.type = FFI_TYPE_POINTER;
	cls_pointer_type.elements = NULL;*/

	void*	arg1	= (void*)0x12345678;
	void*	arg2	= (void*)0x89abcdef;
	ffi_arg	res		= 0;

	arg_types[0] = &ffi_type_pointer;
	arg_types[1] = &ffi_type_pointer;
	arg_types[2] = NULL;

	CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &ffi_type_pointer,
		arg_types) == FFI_OK);

	args[0] = &arg1;
	args[1] = &arg2;
	args[2] = NULL;

	ffi_call(&cif, FFI_FN(cls_pointer_fn), &res, args);
	/* { dg-output "0x12345678 0x89abcdef: 0x9be02467" } */
	printf("res: 0x%08x\n", res);
	/* { dg-output "\nres: 0x9be02467" } */

	CHECK(ffi_prep_closure(pcl, &cif, cls_pointer_gn, NULL) == FFI_OK);

	res = (ffi_arg)((void*(*)(void*, void*))(pcl))(arg1, arg2);
	/* { dg-output "\n0x12345678 0x89abcdef: 0x9be02467" } */
	printf("res: 0x%08x\n", res);
	/* { dg-output "\nres: 0x9be02467" } */

	exit(0);
}
