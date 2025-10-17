/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

/* ffi_prep_args is called by the assembly routine once stack
   space has been allocated for the function's arguments.  */

void ffi_prep_args(char *stack, extended_cif *ecif)
{
  unsigned int i;
  int tmp;
  unsigned int avn;
  void **p_argv;
  char *argp;
  ffi_type **p_arg;

  tmp = 0;
  argp = stack;

  if (ecif->cif->rtype->type == FFI_TYPE_STRUCT && ecif->cif->rtype->size > 8)
    {
      *(void **) argp = ecif->rvalue;
      argp += 4;
    }

  avn = ecif->cif->nargs;
  p_argv = ecif->avalue;

  for (i = ecif->cif->nargs, p_arg = ecif->cif->arg_types;
       (i != 0) && (avn != 0);
       i--, p_arg++)
    {
      size_t z;

      /* Align if necessary.  */
      if (((*p_arg)->alignment - 1) & (unsigned) argp)
	argp = (char *) FFI_ALIGN (argp, (*p_arg)->alignment);

      if (avn != 0) 
	{
	  avn--;
	  z = (*p_arg)->size;
	  if (z < sizeof (int))
	    {
	      z = sizeof (int);

	      switch ((*p_arg)->type)
		{
		case FFI_TYPE_SINT8:
		  *(signed int *) argp = (signed int)*(SINT8 *)(* p_argv);
		  break;
		  
		case FFI_TYPE_UINT8:
		  *(unsigned int *) argp = (unsigned int)*(UINT8 *)(* p_argv);
		  break;
		  
		case FFI_TYPE_SINT16:
		  *(signed int *) argp = (signed int)*(SINT16 *)(* p_argv);
		  break;
		  
		case FFI_TYPE_UINT16:
		  *(unsigned int *) argp = (unsigned int)*(UINT16 *)(* p_argv);
		  break;
		  
		case FFI_TYPE_STRUCT:
	  	  z = (*p_arg)->size;
	  	  if ((*p_arg)->alignment != 1)
		    memcpy (argp, *p_argv, z);
		  else
		    memcpy (argp + 4 - z, *p_argv, z);
	  	  z = sizeof (int);
		  break;

		default:
		  FFI_ASSERT(0);
		}
	    }
	  else if (z == sizeof (int))
	    {
	       *(unsigned int *) argp = (unsigned int)*(UINT32 *)(* p_argv);
	    }
	  else
	    {
	      if ((*p_arg)->type == FFI_TYPE_STRUCT)
	        {
		  if (z > 8)
		    {
		      *(unsigned int *) argp = (unsigned int)(void *)(* p_argv);
		      z = sizeof(void *);
		    }
		  else
		    {
	              memcpy(argp, *p_argv, z);
		      z = 8;
		    }
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
    }
  
  return;
}

/* Perform machine dependent cif processing.  */
ffi_status
ffi_prep_cif_machdep(ffi_cif *cif)
{
  /* Set the return type flag.  */
  switch (cif->rtype->type)
    {
    case FFI_TYPE_VOID:
      cif->flags = (unsigned) cif->rtype->type;
      break;

    case FFI_TYPE_STRUCT:
      if (cif->rtype->size <= 4)
	cif->flags = FFI_TYPE_INT;

      else if (cif->rtype->size <= 8)
	cif->flags = FFI_TYPE_DOUBLE;

      else
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

extern void ffi_call_SYSV(void (*)(char *, extended_cif *), extended_cif *,
			  unsigned, unsigned, unsigned *, void (*fn)(void));

void ffi_call(ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue)
{
  extended_cif ecif;

  ecif.cif = cif;
  ecif.avalue = avalue;
  
  /* If the return value is a struct and we don't have
     a return value address then we need to make one.  */
  if ((rvalue == NULL) && 
      (cif->rtype->type == FFI_TYPE_STRUCT))
    {
      ecif.rvalue = alloca (cif->rtype->size);
    }
  else
    ecif.rvalue = rvalue;    
  
  switch (cif->abi) 
    {
    case FFI_SYSV:
      ffi_call_SYSV(ffi_prep_args, &ecif, cif->bytes, 
		    cif->flags, ecif.rvalue, fn);
      if (cif->rtype->type == FFI_TYPE_STRUCT)
	{
	  int size = cif->rtype->size;
	  int align = cif->rtype->alignment;

	  if (size < 4)
	    {
	      if (align == 1)
	        *(unsigned long *)(ecif.rvalue) <<= (4 - size) * 8;
	    }
	  else if (4 < size && size < 8)
	    {
	      if (align == 1)
		{
		  memcpy (ecif.rvalue, ecif.rvalue + 8-size, size);
		}
	      else if (align == 2)
		{
		  if (size & 1)
		    size += 1;

		  if (size != 8)
		    memcpy (ecif.rvalue, ecif.rvalue + 8-size, size);
		}
	    }
	}
      break;

    default:
      FFI_ASSERT(0);
      break;
    }
}

