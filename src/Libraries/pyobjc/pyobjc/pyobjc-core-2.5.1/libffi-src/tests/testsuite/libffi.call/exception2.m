/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#import <Foundation/Foundation.h>

static void
closure_raise(ffi_cif* cif,void* resp,void** args, void* userdata)
{
  [NSException raise:NSInvalidArgumentException format:@"Dummy exception"];
}


typedef void (*testfunc)(void);

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  int ok;

#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;

#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif

 NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];





  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 0, 
		     &ffi_type_void, args) == FFI_OK);
  
  CHECK(ffi_prep_closure(pcl, &cif, closure_raise,
                         (void *) 3 /* userdata */) == FFI_OK);
  ok = 0;
  NS_DURING
	((testfunc)pcl)();
  NS_HANDLER
 	ok = 1;

  NS_ENDHANDLER
  
  CHECK(ok);

  [pool release];
  exit(0);
}
