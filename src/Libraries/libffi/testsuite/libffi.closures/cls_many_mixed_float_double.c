/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
#include <float.h>
#include <math.h>

#define NARGS 16

static void cls_mixed_float_double_fn(ffi_cif* cif , void* ret, void** args,
			      void* userdata __UNUSED__)
{
    double r = 0;
    unsigned int i;
    double t;
    for(i=0; i < cif->nargs; i++)
    {
        if(cif->arg_types[i] == &ffi_type_double) {
				t = *(((double**)(args))[i]);
        } else {
				t = *(((float**)(args))[i]);
        }
        r += t;
    }
    *((double*)ret) = r;
}
typedef double (*cls_mixed)(double, float, double, double, double, double, double, float, float, double, float, float);

int main (void)
{
    ffi_cif cif;
    ffi_closure *closure;
	void* code;
    ffi_type *argtypes[12] = {&ffi_type_double, &ffi_type_float, &ffi_type_double,
                          &ffi_type_double, &ffi_type_double, &ffi_type_double,
                          &ffi_type_double, &ffi_type_float, &ffi_type_float,
                          &ffi_type_double, &ffi_type_float, &ffi_type_float};


    closure = ffi_closure_alloc(sizeof(ffi_closure), (void**)&code);
    if(closure ==NULL)
		abort();
    CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 12, &ffi_type_double, argtypes) == FFI_OK);
	CHECK(ffi_prep_closure_loc(closure, &cif, cls_mixed_float_double_fn, NULL,  code) == FFI_OK);
    double ret = ((cls_mixed)code)(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2);
    ffi_closure_free(closure);
	if(fabs(ret - 7.8) < FLT_EPSILON)
		exit(0);
	else
		abort();
}

