/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#ifndef SVWINFNT_H_
#define SVWINFNT_H_

#include FT_INTERNAL_SERVICE_H
#include FT_WINFONTS_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_WINFNT  "winfonts"

  typedef FT_Error
  (*FT_WinFnt_GetHeaderFunc)( FT_Face               face,
                              FT_WinFNT_HeaderRec  *aheader );


  FT_DEFINE_SERVICE( WinFnt )
  {
    FT_WinFnt_GetHeaderFunc  get_header;
  };

  /* */


FT_END_HEADER


#endif /* SVWINFNT_H_ */


/* END */
