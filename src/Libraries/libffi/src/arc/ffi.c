/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#include <stdint.h>

#include <sys/cachectl.h>

/* for little endian ARC, the code is in fact stored as mixed endian for
   performance reasons */
#if __BIG_ENDIAN__
#define CODE_ENDIAN(x) (x)
#else
#define CODE_ENDIAN(x) ( (((uint32_t) (x)) << 16) | (((uint32_t) (x)) >> 16))
#endif

/* ffi_prep_args is called by the assembly routine once stack
   space has been allocated for the function's arguments.  */

void
ffi_prep_args (char *stack, extended_cif * ecif)
{
  unsigned int i;
  void **p_argv;
  char *argp;
  ffi_type **p_arg;

  argp = stack;

  if (ecif->cif->rtype->type == FFI_TYPE_STRUCT)
    {
      *(void **) argp = ecif->rvalue;
      argp += 4;
    }

  p_argv = ecif->avalue;

  for (i = ecif->cif->nargs, p_arg = ecif->cif->arg_types;
       (i != 0); i--, p_arg++)
    {
      size_t z;
      int alignment;

      /* align alignment to 4 */
      alignment = (((*p_arg)->alignment - 1) | 3) + 1;

      /* Align if necessary.  */
      if ((alignment - 1) & (unsigned) argp)
	argp = (char *) FFI_ALIGN (argp, alignment);

      z = (*p_arg)->size;
      if (z < sizeof (int))
	{
	  z = sizeof (int);

	  switch ((*p_arg)->type)
	    {
	    case FFI_TYPE_SINT8:
	      *(signed int *) argp = (signed int) *(SINT8 *) (*p_argv);
	      break;

	    case FFI_TYPE_UINT8:
	      *(unsigned int *) argp = (unsigned int) *(UINT8 *) (*p_argv);
	      break;

	    case FFI_TYPE_SINT16:
	      *(signed int *) argp = (signed int) *(SINT16 *) (*p_argv);
	      break;

	    case FFI_TYPE_UINT16:
	      *(unsigned int *) argp = (unsigned int) *(UINT16 *) (*p_argv);
	      break;

	    case FFI_TYPE_STRUCT:
	      memcpy (argp, *p_argv, (*p_arg)->size);
	      break;

	    default:
	      FFI_ASSERT (0);
	    }
	}
      else if (z == sizeof (int))
	{
	  *(unsigned int *) argp = (unsigned int) *(UINT32 *) (*p_argv);
	}
      else
	{
	  if ((*p_arg)->type == FFI_TYPE_STRUCT)
	    {
	      memcpy (argp, *p_argv, z);
	    }
	  else
	    {
	      /* Double or long long 64bit.  */
	      memcpy (argp, *p_argv, z);
	    }
	}
      p_argv++;
      argp += z;
    }

  return;
}

/* Perform machine dependent cif processing.  */
ffi_status
ffi_prep_cif_machdep (ffi_cif * cif)
{
  /* Set the return type flag.  */
  switch (cif->rtype->type)
    {
    case FFI_TYPE_VOID:
      cif->flags = (unsigned) cif->rtype->type;
      break;

    case FFI_TYPE_STRUCT:
      cif->flags = (unsigned) cif->rtype->type;
      break;

    case FFI_TYPE_SINT64:
    case FFI_TYPE_UINT64:
    case FFI_TYPE_DOUBLE:
      cif->flags = FFI_TYPE_DOUBLE;
      break;

    case FFI_TYPE_FLOAT:
    default:
      cif->flags = FFI_TYPE_INT;
      break;
    }

  return FFI_OK;
}

extern void ffi_call_ARCompact (void (*)(char *, extended_cif *),
				extended_cif *, unsigned, unsigned,
				unsigned *, void (*fn) (void));

void
ffi_call (ffi_cif * cif, void (*fn) (void), void *rvalue, void **avalue)
{
  extended_cif ecif;

  ecif.cif = cif;
  ecif.avalue = avalue;

  /* If the return value is a struct and we don't have
     a return value address then we need to make one.  */
  if ((rvalue == NULL) && (cif->rtype->type == FFI_TYPE_STRUCT))
    {
      ecif.rvalue = alloca (cif->rtype->size);
    }
  else
    ecif.rvalue = rvalue;

  switch (cif->abi)
    {
    case FFI_ARCOMPACT:
      ffi_call_ARCompact (ffi_prep_args, &ecif, cif->bytes,
			  cif->flags, ecif.rvalue, fn);
      break;

    default:
      FFI_ASSERT (0);
      break;
    }
}

int
ffi_closure_inner_ARCompact (ffi_closure * closure, void *rvalue,
			     ffi_arg * args)
{
  void **arg_area, **p_argv;
  ffi_cif *cif = closure->cif;
  char *argp = (char *) args;
  ffi_type **p_argt;
  int i;

  arg_area = (void **) alloca (cif->nargs * sizeof (void *));

  /* handle hidden argument */
  if (cif->flags == FFI_TYPE_STRUCT)
    {
      rvalue = *(void **) argp;
      argp += 4;
    }

  p_argv = arg_area;

  for (i = 0, p_argt = cif->arg_types; i < cif->nargs;
       i++, p_argt++, p_argv++)
    {
      size_t z;
      int alignment;

      /* align alignment to 4 */
      alignment = (((*p_argt)->alignment - 1) | 3) + 1;

      /* Align if necessary.  */
      if ((alignment - 1) & (unsigned) argp)
	argp = (char *) FFI_ALIGN (argp, alignment);

      z = (*p_argt)->size;
      *p_argv = (void *) argp;
      argp += z;
    }

  (closure->fun) (cif, rvalue, arg_area, closure->user_data);

  return cif->flags;
}

extern void ffi_closure_ARCompact (void);

ffi_status
ffi_prep_closure_loc (ffi_closure * closure, ffi_cif * cif,
		      void (*fun) (ffi_cif *, void *, void **, void *),
		      void *user_data, void *codeloc)
{
  uint32_t *tramp = (uint32_t *) & (closure->tramp[0]);

  switch (cif->abi)
    {
    case FFI_ARCOMPACT:
      FFI_ASSERT (tramp == codeloc);
      tramp[0] = CODE_ENDIAN (0x200a1fc0);	/* mov r8, pcl  */
      tramp[1] = CODE_ENDIAN (0x20200f80);	/* j [long imm] */
      tramp[2] = CODE_ENDIAN (ffi_closure_ARCompact);
      break;

    default:
      return FFI_BAD_ABI;
    }

  closure->cif = cif;
  closure->fun = fun;
  closure->user_data = user_data;
  cacheflush (codeloc, FFI_TRAMPOLINE_SIZE, BCACHE);

  return FFI_OK;
}

