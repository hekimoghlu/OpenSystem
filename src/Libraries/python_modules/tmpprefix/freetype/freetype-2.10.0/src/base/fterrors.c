/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#include <ft2build.h>
#include FT_ERRORS_H


  /* documentation is in fterrors.h */

  FT_EXPORT_DEF( const char* )
  FT_Error_String( FT_Error  error_code )
  {
    if ( error_code <  0                                ||
         error_code >= FT_ERR_CAT( FT_ERR_PREFIX, Max ) )
      return NULL;

#if defined( FT_CONFIG_OPTION_ERROR_STRINGS ) || \
    defined( FT_DEBUG_LEVEL_ERROR )

#undef FTERRORS_H_
#define FT_ERROR_START_LIST     switch ( FT_ERROR_BASE( error_code ) ) {
#define FT_ERRORDEF( e, v, s )    case v: return s;
#define FT_ERROR_END_LIST       }

#include FT_ERRORS_H

#endif /* defined( FT_CONFIG_OPTION_ERROR_STRINGS ) || ... */

    return NULL;
  }
