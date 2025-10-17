/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#include "ffi.h"
#include "ffi_common.h"
#include "ffi_powerpc.h"

#if HAVE_LONG_DOUBLE_VARIANT
/* Adjust ffi_type_longdouble.  */
void FFI_HIDDEN
ffi_prep_types (ffi_abi abi)
{
# if FFI_TYPE_LONGDOUBLE != FFI_TYPE_DOUBLE
#  ifdef POWERPC64
  ffi_prep_types_linux64 (abi);
#  else
  ffi_prep_types_sysv (abi);
#  endif
# endif
}
#endif

/* Perform machine dependent cif processing */
ffi_status FFI_HIDDEN
ffi_prep_cif_machdep (ffi_cif *cif)
{
#ifdef POWERPC64
  return ffi_prep_cif_linux64 (cif);
#else
  return ffi_prep_cif_sysv (cif);
#endif
}

ffi_status FFI_HIDDEN
ffi_prep_cif_machdep_var (ffi_cif *cif,
			  unsigned int nfixedargs MAYBE_UNUSED,
			  unsigned int ntotalargs MAYBE_UNUSED)
{
#ifdef POWERPC64
  return ffi_prep_cif_linux64_var (cif, nfixedargs, ntotalargs);
#else
  return ffi_prep_cif_sysv (cif);
#endif
}

void
ffi_call(ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue)
{
  /* The final SYSV ABI says that structures smaller or equal 8 bytes
     are returned in r3/r4.  A draft ABI used by linux instead returns
     them in memory.

     We bounce-buffer SYSV small struct return values so that sysv.S
     can write r3 and r4 to memory without worrying about struct size.
   
     For ELFv2 ABI, use a bounce buffer for homogeneous structs too,
     for similar reasons.  */
  unsigned long smst_buffer[8];
  extended_cif ecif;

  ecif.cif = cif;
  ecif.avalue = avalue;

  ecif.rvalue = rvalue;
  if ((cif->flags & FLAG_RETURNS_SMST) != 0)
    ecif.rvalue = smst_buffer;
  /* Ensure that we have a valid struct return value.
     FIXME: Isn't this just papering over a user problem?  */
  else if (!rvalue && cif->rtype->type == FFI_TYPE_STRUCT)
    ecif.rvalue = alloca (cif->rtype->size);

#ifdef POWERPC64
  ffi_call_LINUX64 (&ecif, -(long) cif->bytes, cif->flags, ecif.rvalue, fn);
#else
  ffi_call_SYSV (&ecif, -cif->bytes, cif->flags, ecif.rvalue, fn);
#endif

  /* Check for a bounce-buffered return value */
  if (rvalue && ecif.rvalue == smst_buffer)
    {
      unsigned int rsize = cif->rtype->size;
#ifndef __LITTLE_ENDIAN__
      /* The SYSV ABI returns a structure of up to 4 bytes in size
	 left-padded in r3.  */
# ifndef POWERPC64
      if (rsize <= 4)
	memcpy (rvalue, (char *) smst_buffer + 4 - rsize, rsize);
      else
# endif
	/* The SYSV ABI returns a structure of up to 8 bytes in size
	   left-padded in r3/r4, and the ELFv2 ABI similarly returns a
	   structure of up to 8 bytes in size left-padded in r3.  */
	if (rsize <= 8)
	  memcpy (rvalue, (char *) smst_buffer + 8 - rsize, rsize);
	else
#endif
	  memcpy (rvalue, smst_buffer, rsize);
    }
}


ffi_status
ffi_prep_closure_loc (ffi_closure *closure,
		      ffi_cif *cif,
		      void (*fun) (ffi_cif *, void *, void **, void *),
		      void *user_data,
		      void *codeloc)
{
#ifdef POWERPC64
  return ffi_prep_closure_loc_linux64 (closure, cif, fun, user_data, codeloc);
#else
  return ffi_prep_closure_loc_sysv (closure, cif, fun, user_data, codeloc);
#endif
}
