/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#ifndef PSARRST_H_
#define PSARRST_H_


#include "pserror.h"


FT_BEGIN_HEADER


  /* need to define the struct here (not opaque) so it can be allocated by */
  /* clients                                                               */
  typedef struct  CF2_ArrStackRec_
  {
    FT_Memory  memory;
    FT_Error*  error;

    size_t  sizeItem;       /* bytes per element             */
    size_t  allocated;      /* items allocated               */
    size_t  chunk;          /* allocation increment in items */
    size_t  count;          /* number of elements allocated  */
    size_t  totalSize;      /* total bytes allocated         */

    void*  ptr;             /* ptr to data                   */

  } CF2_ArrStackRec, *CF2_ArrStack;


  FT_LOCAL( void )
  cf2_arrstack_init( CF2_ArrStack  arrstack,
                     FT_Memory     memory,
                     FT_Error*     error,
                     size_t        sizeItem );
  FT_LOCAL( void )
  cf2_arrstack_finalize( CF2_ArrStack  arrstack );

  FT_LOCAL( void )
  cf2_arrstack_setCount( CF2_ArrStack  arrstack,
                         size_t        numElements );
  FT_LOCAL( void )
  cf2_arrstack_clear( CF2_ArrStack  arrstack );
  FT_LOCAL( size_t )
  cf2_arrstack_size( const CF2_ArrStack  arrstack );

  FT_LOCAL( void* )
  cf2_arrstack_getBuffer( const CF2_ArrStack  arrstack );
  FT_LOCAL( void* )
  cf2_arrstack_getPointer( const CF2_ArrStack  arrstack,
                           size_t              idx );

  FT_LOCAL( void )
  cf2_arrstack_push( CF2_ArrStack  arrstack,
                     const void*   ptr );


FT_END_HEADER


#endif /* PSARRST_H_ */


/* END */
