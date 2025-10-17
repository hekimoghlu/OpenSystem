/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#include <ffi.h>
#include <ffi_common.h>
#include <stdlib.h>

/* Round up to FFI_SIZEOF_ARG. */

#define STACK_ARG_SIZE(x) ALIGN(x, FFI_SIZEOF_ARG)

/* Perform machine independent initialization of aggregate type
   specifications. */

static ffi_status initialize_aggregate(ffi_type *arg)
{
  ffi_type **ptr;

  if (UNLIKELY(arg == NULL || arg->elements == NULL))
    return FFI_BAD_TYPEDEF;

  arg->size = 0;
  arg->alignment = 0;

  ptr = &(arg->elements[0]);

  if (UNLIKELY(ptr == 0))
    return FFI_BAD_TYPEDEF;

  while ((*ptr) != NULL)
    {
      if (UNLIKELY(((*ptr)->size == 0)
		    && (initialize_aggregate((*ptr)) != FFI_OK)))
	return FFI_BAD_TYPEDEF;

      /* Perform a sanity check on the argument type */
      FFI_ASSERT_VALID_TYPE(*ptr);

      arg->size = ALIGN(arg->size, (*ptr)->alignment);
      arg->size += (*ptr)->size;

      arg->alignment = (arg->alignment > (*ptr)->alignment) ?
	arg->alignment : (*ptr)->alignment;

      ptr++;
    }

  /* Structure size includes tail padding.  This is important for
     structures that fit in one register on ABIs like the PowerPC64
     Linux ABI that right justify small structs in a register.
     It's also needed for nested structure layout, for example
     struct A { long a; char b; }; struct B { struct A x; char y; };
     should find y at an offset of 2*sizeof(long) and result in a
     total size of 3*sizeof(long).  */
  arg->size = ALIGN (arg->size, arg->alignment);

  /* On some targets, the ABI defines that structures have an additional
     alignment beyond the "natural" one based on their elements.  */
#ifdef FFI_AGGREGATE_ALIGNMENT
  if (FFI_AGGREGATE_ALIGNMENT > arg->alignment)
    arg->alignment = FFI_AGGREGATE_ALIGNMENT;
#endif

  if (arg->size == 0)
    return FFI_BAD_TYPEDEF;
  else
    return FFI_OK;
}

#ifndef __CRIS__
/* The CRIS ABI specifies structure elements to have byte
   alignment only, so it completely overrides this functions,
   which assumes "natural" alignment and padding.  */

/* Perform machine independent ffi_cif preparation, then call
   machine dependent routine. */

/* For non variadic functions isvariadic should be 0 and
   nfixedargs==ntotalargs.

   For variadic calls, isvariadic should be 1 and nfixedargs
   and ntotalargs set as appropriate. nfixedargs must always be >=1 */


ffi_status FFI_HIDDEN ffi_prep_cif_core(ffi_cif *cif, ffi_abi abi,
			     unsigned int isvariadic,
                             unsigned int nfixedargs,
                             unsigned int ntotalargs,
			     ffi_type *rtype, ffi_type **atypes)
{
  unsigned bytes = 0;
  unsigned int i;
  ffi_type **ptr;

  FFI_ASSERT(cif != NULL);
  FFI_ASSERT((!isvariadic) || (nfixedargs >= 1));
  FFI_ASSERT(nfixedargs <= ntotalargs);

  if (! (abi > FFI_FIRST_ABI && abi < FFI_LAST_ABI))
    return FFI_BAD_ABI;

  cif->abi = abi;
  cif->arg_types = atypes;
  cif->nargs = ntotalargs;
  cif->rtype = rtype;

  cif->flags = 0;

#if HAVE_LONG_DOUBLE_VARIANT
  ffi_prep_types (abi);
#endif

  /* Initialize the return type if necessary */
  if ((cif->rtype->size == 0) && (initialize_aggregate(cif->rtype) != FFI_OK))
    return FFI_BAD_TYPEDEF;

#ifndef FFI_TARGET_HAS_COMPLEX_TYPE
  if (rtype->type == FFI_TYPE_COMPLEX)
    abort();
#endif
  /* Perform a sanity check on the return type */
  FFI_ASSERT_VALID_TYPE(cif->rtype);

  /* x86, x86-64 and s390 stack space allocation is handled in prep_machdep. */
#if !defined FFI_TARGET_SPECIFIC_STACK_SPACE_ALLOCATION
  /* Make space for the return structure pointer */
  if (cif->rtype->type == FFI_TYPE_STRUCT
#ifdef SPARC
      && (cif->abi != FFI_V9 || cif->rtype->size > 32)
#endif
#ifdef TILE
      && (cif->rtype->size > 10 * FFI_SIZEOF_ARG)
#endif
#ifdef XTENSA
      && (cif->rtype->size > 16)
#endif
#ifdef NIOS2
      && (cif->rtype->size > 8)
#endif
     )
    bytes = STACK_ARG_SIZE(sizeof(void*));
#endif

  for (ptr = cif->arg_types, i = cif->nargs; i > 0; i--, ptr++)
    {

      /* Initialize any uninitialized aggregate type definitions */
      if (((*ptr)->size == 0) && (initialize_aggregate((*ptr)) != FFI_OK))
	return FFI_BAD_TYPEDEF;

#ifndef FFI_TARGET_HAS_COMPLEX_TYPE
      if ((*ptr)->type == FFI_TYPE_COMPLEX)
	abort();
#endif
      /* Perform a sanity check on the argument type, do this
	 check after the initialization.  */
      FFI_ASSERT_VALID_TYPE(*ptr);

#if !defined FFI_TARGET_SPECIFIC_STACK_SPACE_ALLOCATION
#ifdef SPARC
      if (((*ptr)->type == FFI_TYPE_STRUCT
	   && ((*ptr)->size > 16 || cif->abi != FFI_V9))
	  || ((*ptr)->type == FFI_TYPE_LONGDOUBLE
	      && cif->abi != FFI_V9))
	bytes += sizeof(void*);
      else
#endif
	{
	  /* Add any padding if necessary */
	  if (((*ptr)->alignment - 1) & bytes)
	    bytes = (unsigned)ALIGN(bytes, (*ptr)->alignment);

#ifdef TILE
	  if (bytes < 10 * FFI_SIZEOF_ARG &&
	      bytes + STACK_ARG_SIZE((*ptr)->size) > 10 * FFI_SIZEOF_ARG)
	    {
	      /* An argument is never split between the 10 parameter
		 registers and the stack.  */
	      bytes = 10 * FFI_SIZEOF_ARG;
	    }
#endif
#ifdef XTENSA
	  if (bytes <= 6*4 && bytes + STACK_ARG_SIZE((*ptr)->size) > 6*4)
	    bytes = 6*4;
#endif

	  bytes += STACK_ARG_SIZE((*ptr)->size);
	}
#endif
    }

  cif->bytes = bytes;

  /* Perform machine dependent cif processing */
#ifdef FFI_TARGET_SPECIFIC_VARIADIC
  if (isvariadic)
	return ffi_prep_cif_machdep_var(cif, nfixedargs, ntotalargs);
#endif

  return ffi_prep_cif_machdep(cif);
}
#endif /* not __CRIS__ */

ffi_status ffi_prep_cif(ffi_cif *cif, ffi_abi abi, unsigned int nargs,
			     ffi_type *rtype, ffi_type **atypes)
{
  return ffi_prep_cif_core(cif, abi, 0, nargs, nargs, rtype, atypes);
}

ffi_status ffi_prep_cif_var(ffi_cif *cif,
                            ffi_abi abi,
                            unsigned int nfixedargs,
                            unsigned int ntotalargs,
                            ffi_type *rtype,
                            ffi_type **atypes)
{
  return ffi_prep_cif_core(cif, abi, 1, nfixedargs, ntotalargs, rtype, atypes);
}

#if FFI_CLOSURES

ffi_status
ffi_prep_closure (ffi_closure* closure,
		  ffi_cif* cif,
		  void (*fun)(ffi_cif*,void*,void**,void*),
		  void *user_data)
{
  return ffi_prep_closure_loc (closure, cif, fun, user_data, closure);
}

#endif
