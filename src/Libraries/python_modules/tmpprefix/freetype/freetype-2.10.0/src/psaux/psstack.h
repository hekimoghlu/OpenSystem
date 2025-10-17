/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
#ifndef PSSTACK_H_
#define PSSTACK_H_


FT_BEGIN_HEADER


  /* CFF operand stack; specified maximum of 48 or 192 values */
  typedef struct  CF2_StackNumber_
  {
    union
    {
      CF2_Fixed  r;      /* 16.16 fixed point */
      CF2_Frac   f;      /* 2.30 fixed point (for font matrix) */
      CF2_Int    i;
    } u;

    CF2_NumberType  type;

  } CF2_StackNumber;


  typedef struct  CF2_StackRec_
  {
    FT_Memory         memory;
    FT_Error*         error;
    CF2_StackNumber*  buffer;
    CF2_StackNumber*  top;
    FT_UInt           stackSize;

  } CF2_StackRec, *CF2_Stack;


  FT_LOCAL( CF2_Stack )
  cf2_stack_init( FT_Memory  memory,
                  FT_Error*  error,
                  FT_UInt    stackSize );
  FT_LOCAL( void )
  cf2_stack_free( CF2_Stack  stack );

  FT_LOCAL( CF2_UInt )
  cf2_stack_count( CF2_Stack  stack );

  FT_LOCAL( void )
  cf2_stack_pushInt( CF2_Stack  stack,
                     CF2_Int    val );
  FT_LOCAL( void )
  cf2_stack_pushFixed( CF2_Stack  stack,
                       CF2_Fixed  val );

  FT_LOCAL( CF2_Int )
  cf2_stack_popInt( CF2_Stack  stack );
  FT_LOCAL( CF2_Fixed )
  cf2_stack_popFixed( CF2_Stack  stack );

  FT_LOCAL( CF2_Fixed )
  cf2_stack_getReal( CF2_Stack  stack,
                     CF2_UInt   idx );
  FT_LOCAL( void )
  cf2_stack_setReal( CF2_Stack  stack,
                     CF2_UInt   idx,
                     CF2_Fixed  val );

  FT_LOCAL( void )
  cf2_stack_pop( CF2_Stack  stack,
                 CF2_UInt   num );

  FT_LOCAL( void )
  cf2_stack_roll( CF2_Stack  stack,
                  CF2_Int    count,
                  CF2_Int    idx );

  FT_LOCAL( void )
  cf2_stack_clear( CF2_Stack  stack );


FT_END_HEADER


#endif /* PSSTACK_H_ */


/* END */
