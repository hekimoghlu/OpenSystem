/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#ifndef PSERROR_H_
#define PSERROR_H_


#include FT_MODULE_ERRORS_H

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  CF2_Err_
#define FT_ERR_BASE    FT_Mod_Err_CF2


#include FT_ERRORS_H
#include "psft.h"


FT_BEGIN_HEADER


  /*
   * A poor-man error facility.
   *
   * This code being written in vanilla C, doesn't have the luxury of a
   * language-supported exception mechanism such as the one available in
   * Java.  Instead, we are stuck with using error codes that must be
   * carefully managed and preserved.  However, it is convenient for us to
   * model our error mechanism on a Java-like exception mechanism.
   * When we assign an error code we are thus `throwing' an error.
   *
   * The preservation of an error code is done by coding convention.
   * Upon a function call if the error code is anything other than
   * `FT_Err_Ok', which is guaranteed to be zero, we
   * will return without altering that error.  This will allow the
   * error to propagate and be handled at the appropriate location in
   * the code.
   *
   * This allows a style of code where the error code is initialized
   * up front and a block of calls are made with the error code only
   * being checked after the block.  If a new error occurs, the original
   * error will be preserved and a functional no-op should result in any
   * subsequent function that has an initial error code not equal to
   * `FT_Err_Ok'.
   *
   * Errors are encoded by calling the `FT_THROW' macro.  For example,
   *
   * {
   *   FT_Error  e;
   *
   *
   *   ...
   *   e = FT_THROW( Out_Of_Memory );
   * }
   *
   */


  /* Set error code to a particular value. */
  FT_LOCAL( void )
  cf2_setError( FT_Error*  error,
                FT_Error   value );


  /*
   * A macro that conditionally sets an error code.
   *
   * This macro will first check whether `error' is set;
   * if not, it will set it to `e'.
   *
  */
#define CF2_SET_ERROR( error, e )              \
          cf2_setError( error, FT_THROW( e ) )


FT_END_HEADER


#endif /* PSERROR_H_ */


/* END */
