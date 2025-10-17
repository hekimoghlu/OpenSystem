/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
enum {
  /* The assembly depends on these exact flags.  */
  /* These go in cr7 */
  FLAG_RETURNS_SMST	= 1 << (31-31), /* Used for FFI_SYSV small structs.  */
  FLAG_RETURNS_NOTHING  = 1 << (31-30),
  FLAG_RETURNS_FP       = 1 << (31-29),
  FLAG_RETURNS_64BITS   = 1 << (31-28),

  /* This goes in cr6 */
  FLAG_RETURNS_128BITS  = 1 << (31-27),

  FLAG_COMPAT		= 1 << (31- 8), /* Not used by assembly */

  /* These go in cr1 */
  FLAG_ARG_NEEDS_COPY   = 1 << (31- 7), /* Used by sysv code */
  FLAG_ARG_NEEDS_PSAVE  = FLAG_ARG_NEEDS_COPY, /* Used by linux64 code */
  FLAG_FP_ARGUMENTS     = 1 << (31- 6), /* cr1.eq; specified by ABI */
  FLAG_4_GPR_ARGUMENTS  = 1 << (31- 5),
  FLAG_RETVAL_REFERENCE = 1 << (31- 4)
};

typedef union
{
  float f;
  double d;
} ffi_dblfl;

void FFI_HIDDEN ffi_closure_SYSV (void);
void FFI_HIDDEN ffi_call_SYSV(extended_cif *, unsigned, unsigned, unsigned *,
			      void (*)(void));

void FFI_HIDDEN ffi_prep_types_sysv (ffi_abi);
ffi_status FFI_HIDDEN ffi_prep_cif_sysv (ffi_cif *);
int FFI_HIDDEN ffi_closure_helper_SYSV (ffi_closure *, void *, unsigned long *,
					ffi_dblfl *, unsigned long *);

void FFI_HIDDEN ffi_call_LINUX64(extended_cif *, unsigned long, unsigned long,
				 unsigned long *, void (*)(void));
void FFI_HIDDEN ffi_closure_LINUX64 (void);

void FFI_HIDDEN ffi_prep_types_linux64 (ffi_abi);
ffi_status FFI_HIDDEN ffi_prep_cif_linux64 (ffi_cif *);
ffi_status FFI_HIDDEN ffi_prep_cif_linux64_var (ffi_cif *, unsigned int,
						unsigned int);
void FFI_HIDDEN ffi_prep_args64 (extended_cif *, unsigned long *const);
int FFI_HIDDEN ffi_closure_helper_LINUX64 (ffi_closure *, void *,
					   unsigned long *, ffi_dblfl *);
