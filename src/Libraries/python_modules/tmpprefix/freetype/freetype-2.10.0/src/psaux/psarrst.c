/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "psft.h"
#include FT_INTERNAL_DEBUG_H

#include "psglue.h"
#include "psarrst.h"

#include "pserror.h"


  /*
   * CF2_ArrStack uses an error pointer, to enable shared errors.
   * Shared errors are necessary when multiple objects allow the program
   * to continue after detecting errors.  Only the first error should be
   * recorded.
   */

  FT_LOCAL_DEF( void )
  cf2_arrstack_init( CF2_ArrStack  arrstack,
                     FT_Memory     memory,
                     FT_Error*     error,
                     size_t        sizeItem )
  {
    FT_ASSERT( arrstack );

    /* initialize the structure */
    arrstack->memory    = memory;
    arrstack->error     = error;
    arrstack->sizeItem  = sizeItem;
    arrstack->allocated = 0;
    arrstack->chunk     = 10;    /* chunks of 10 items */
    arrstack->count     = 0;
    arrstack->totalSize = 0;
    arrstack->ptr       = NULL;
  }


  FT_LOCAL_DEF( void )
  cf2_arrstack_finalize( CF2_ArrStack  arrstack )
  {
    FT_Memory  memory = arrstack->memory;     /* for FT_FREE */


    FT_ASSERT( arrstack );

    arrstack->allocated = 0;
    arrstack->count     = 0;
    arrstack->totalSize = 0;

    /* free the data buffer */
    FT_FREE( arrstack->ptr );
  }


  /* allocate or reallocate the buffer size; */
  /* return false on memory error */
  static FT_Bool
  cf2_arrstack_setNumElements( CF2_ArrStack  arrstack,
                               size_t        numElements )
  {
    FT_ASSERT( arrstack );

    {
      FT_Error   error  = FT_Err_Ok;        /* for FT_REALLOC */
      FT_Memory  memory = arrstack->memory; /* for FT_REALLOC */

      size_t  newSize = numElements * arrstack->sizeItem;


      if ( numElements > FT_LONG_MAX / arrstack->sizeItem )
        goto exit;


      FT_ASSERT( newSize > 0 );   /* avoid realloc with zero size */

      if ( !FT_REALLOC( arrstack->ptr, arrstack->totalSize, newSize ) )
      {
        arrstack->allocated = numElements;
        arrstack->totalSize = newSize;

        if ( arrstack->count > numElements )
        {
          /* we truncated the list! */
          CF2_SET_ERROR( arrstack->error, Stack_Overflow );
          arrstack->count = numElements;
          return FALSE;
        }

        return TRUE;     /* success */
      }
    }

  exit:
    /* if there's not already an error, store this one */
    CF2_SET_ERROR( arrstack->error, Out_Of_Memory );

    return FALSE;
  }


  /* set the count, ensuring allocation is sufficient */
  FT_LOCAL_DEF( void )
  cf2_arrstack_setCount( CF2_ArrStack  arrstack,
                         size_t        numElements )
  {
    FT_ASSERT( arrstack );

    if ( numElements > arrstack->allocated )
    {
      /* expand the allocation first */
      if ( !cf2_arrstack_setNumElements( arrstack, numElements ) )
        return;
    }

    arrstack->count = numElements;
  }


  /* clear the count */
  FT_LOCAL_DEF( void )
  cf2_arrstack_clear( CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    arrstack->count = 0;
  }


  /* current number of items */
  FT_LOCAL_DEF( size_t )
  cf2_arrstack_size( const CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    return arrstack->count;
  }


  FT_LOCAL_DEF( void* )
  cf2_arrstack_getBuffer( const CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    return arrstack->ptr;
  }


  /* return pointer to the given element */
  FT_LOCAL_DEF( void* )
  cf2_arrstack_getPointer( const CF2_ArrStack  arrstack,
                           size_t              idx )
  {
    void*  newPtr;


    FT_ASSERT( arrstack );

    if ( idx >= arrstack->count )
    {
      /* overflow */
      CF2_SET_ERROR( arrstack->error, Stack_Overflow );
      idx = 0;    /* choose safe default */
    }

    newPtr = (FT_Byte*)arrstack->ptr + idx * arrstack->sizeItem;

    return newPtr;
  }


  /* push (append) an element at the end of the list;         */
  /* return false on memory error                             */
  /* TODO: should there be a length param for extra checking? */
  FT_LOCAL_DEF( void )
  cf2_arrstack_push( CF2_ArrStack  arrstack,
                     const void*   ptr )
  {
    FT_ASSERT( arrstack );

    if ( arrstack->count == arrstack->allocated )
    {
      /* grow the buffer by one chunk */
      if ( !cf2_arrstack_setNumElements(
             arrstack, arrstack->allocated + arrstack->chunk ) )
      {
        /* on error, ignore the push */
        return;
      }
    }

    FT_ASSERT( ptr );

    {
      size_t  offset = arrstack->count * arrstack->sizeItem;
      void*   newPtr = (FT_Byte*)arrstack->ptr + offset;


      FT_MEM_COPY( newPtr, ptr, arrstack->sizeItem );
      arrstack->count += 1;
    }
  }


/* END */
