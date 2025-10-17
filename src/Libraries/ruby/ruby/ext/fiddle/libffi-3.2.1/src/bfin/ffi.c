/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include <stdio.h>

/* Maximum number of GPRs available for argument passing.  */
#define MAX_GPRARGS 3

/*
 * Return types
 */
#define FFIBFIN_RET_VOID 0
#define FFIBFIN_RET_BYTE 1
#define FFIBFIN_RET_HALFWORD 2
#define FFIBFIN_RET_INT64 3
#define FFIBFIN_RET_INT32 4

/*====================================================================*/
/*                          PROTOTYPE          *
 /*====================================================================*/
void ffi_prep_args(unsigned char *, extended_cif *);

/*====================================================================*/
/*                          Externals                                 */
/*                          (Assembly)                                */
/*====================================================================*/

extern void ffi_call_SYSV(unsigned, extended_cif *, void(*)(unsigned char *, extended_cif *), unsigned, void *, void(*fn)(void));

/*====================================================================*/
/*                          Implementation                            */
/*                                                            */
/*====================================================================*/


/*
 * This function calculates the return type (size) based on type.
 */

ffi_status ffi_prep_cif_machdep(ffi_cif *cif)
{
   /* --------------------------------------*
    *   Return handling                *
    * --------------------------------------*/
   switch (cif->rtype->type) {
      case FFI_TYPE_VOID:
         cif->flags = FFIBFIN_RET_VOID;
         break;
      case FFI_TYPE_UINT16:
      case FFI_TYPE_SINT16:
         cif->flags = FFIBFIN_RET_HALFWORD;
         break;
      case FFI_TYPE_UINT8:
         cif->flags = FFIBFIN_RET_BYTE;
         break;
      case FFI_TYPE_INT:
      case FFI_TYPE_UINT32:
      case FFI_TYPE_SINT32:
      case FFI_TYPE_FLOAT:
      case FFI_TYPE_POINTER:
      case FFI_TYPE_SINT8:
         cif->flags = FFIBFIN_RET_INT32;
         break;
      case FFI_TYPE_SINT64:
      case FFI_TYPE_UINT64:
      case FFI_TYPE_DOUBLE:
          cif->flags = FFIBFIN_RET_INT64;
          break;
      case FFI_TYPE_STRUCT:
         if (cif->rtype->size <= 4){
        	 cif->flags = FFIBFIN_RET_INT32;
         }else if (cif->rtype->size == 8){
        	 cif->flags = FFIBFIN_RET_INT64;
         }else{
        	 //it will return via a hidden pointer in P0
        	 cif->flags = FFIBFIN_RET_VOID;
         }
         break;
      default:
         FFI_ASSERT(0);
         break;
   }
   return FFI_OK;
}

/*
 * This will prepare the arguments and will call the assembly routine
 * cif = the call interface
 * fn = the function to be called
 * rvalue = the return value
 * avalue = the arguments
 */
void ffi_call(ffi_cif *cif, void(*fn)(void), void *rvalue, void **avalue)
{
   int ret_type = cif->flags;
   extended_cif ecif;
   ecif.cif = cif;
   ecif.avalue = avalue;
   ecif.rvalue = rvalue;

   switch (cif->abi) {
      case FFI_SYSV:
         ffi_call_SYSV(cif->bytes, &ecif, ffi_prep_args, ret_type, ecif.rvalue, fn);
         break;
      default:
         FFI_ASSERT(0);
         break;
   }
}


/*
* This function prepares the parameters (copies them from the ecif to the stack)
*  to call the function (ffi_prep_args is called by the assembly routine in file
*  sysv.S, which also calls the actual function)
*/
void ffi_prep_args(unsigned char *stack, extended_cif *ecif)
{
   register unsigned int i = 0;
   void **p_argv;
   unsigned char *argp;
   ffi_type **p_arg;
   argp = stack;
   p_argv = ecif->avalue;
   for (i = ecif->cif->nargs, p_arg = ecif->cif->arg_types;
        (i != 0);
        i--, p_arg++) {
      size_t z;
      z = (*p_arg)->size;
      if (z < sizeof(int)) {
         z = sizeof(int);
         switch ((*p_arg)->type) {
            case FFI_TYPE_SINT8: {
                  signed char v = *(SINT8 *)(* p_argv);
                  signed int t = v;
                  *(signed int *) argp = t;
               }
               break;
            case FFI_TYPE_UINT8: {
                  unsigned char v = *(UINT8 *)(* p_argv);
                  unsigned int t = v;
                  *(unsigned int *) argp = t;
               }
               break;
            case FFI_TYPE_SINT16:
               *(signed int *) argp = (signed int) * (SINT16 *)(* p_argv);
               break;
            case FFI_TYPE_UINT16:
               *(unsigned int *) argp = (unsigned int) * (UINT16 *)(* p_argv);
               break;
            case FFI_TYPE_STRUCT:
               memcpy(argp, *p_argv, (*p_arg)->size);
               break;
            default:
               FFI_ASSERT(0);
               break;
         }
      } else if (z == sizeof(int)) {
         *(unsigned int *) argp = (unsigned int) * (UINT32 *)(* p_argv);
      } else {
         memcpy(argp, *p_argv, z);
      }
      p_argv++;
      argp += z;
   }
}



