/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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

static void
cls_double_va_fn(ffi_cif* cif, void* resp, void** args, void* userdata)
{
	char*	format		= *(char**)args[0];
	double	doubleValue	= *(double*)args[1];

	*(ffi_arg*)resp = printf(format, doubleValue);
}

int main (void)
{
	ffi_cif cif;
#ifndef USING_MMAP
	static ffi_closure cl;
#endif
	ffi_closure *pcl;
	void* args[3];
	ffi_type* arg_types[3];

#ifdef USING_MMAP
	pcl = allocate_mmap(sizeof(ffi_closure));
#else
	pcl = &cl;
#endif

	char*	format		= "%.1f\n";
	double	doubleArg	= 7;
	ffi_arg	res			= 0;

	arg_types[0] = &ffi_type_pointer;
	arg_types[1] = &ffi_type_double;
	arg_types[2] = NULL;

	CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &ffi_type_sint,
		arg_types) == FFI_OK);

	args[0] = &format;
	args[1] = &doubleArg;
	args[2] = NULL;

	ffi_call(&cif, FFI_FN(printf), &res, args);
	// { dg-output "7.0" }
	printf("res: %d\n", res);
	// { dg-output "\nres: 4" }

	CHECK(ffi_prep_closure(pcl, &cif, cls_double_va_fn, NULL) == FFI_OK);

	res	= ((int(*)(char*, double))(pcl))(format, doubleArg);
	// { dg-output "\n7.0" }
	printf("res: %d\n", res);
	// { dg-output "\nres: 4" }

	exit(0);
}
