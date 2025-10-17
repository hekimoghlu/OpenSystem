/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
/* { dg-do run } */

#include "ffitest.h"

static void
dummy_fn(ffi_cif* cif __UNUSED__, void* resp __UNUSED__, 
	 void** args __UNUSED__, void* userdata __UNUSED__)
{}

int main (void)
{
	ffi_cif cif;
        void *code;
	ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
	ffi_type* arg_types[1];

	arg_types[0] = NULL;

	CHECK(ffi_prep_cif(&cif, 255, 0, &ffi_type_void,
		arg_types) == FFI_BAD_ABI);

	CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 0, &ffi_type_void,
		arg_types) == FFI_OK);

	cif.abi= 255;

	CHECK(ffi_prep_closure_loc(pcl, &cif, dummy_fn, NULL, code) == FFI_BAD_ABI);

	exit(0);
}

