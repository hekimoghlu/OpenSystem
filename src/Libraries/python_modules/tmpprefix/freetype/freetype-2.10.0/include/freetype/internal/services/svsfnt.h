/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#ifndef SVSFNT_H_
#define SVSFNT_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


  /*
   * SFNT table loading service.
   */

#define FT_SERVICE_ID_SFNT_TABLE  "sfnt-table"


  /*
   * Used to implement FT_Load_Sfnt_Table().
   */
  typedef FT_Error
  (*FT_SFNT_TableLoadFunc)( FT_Face    face,
                            FT_ULong   tag,
                            FT_Long    offset,
                            FT_Byte*   buffer,
                            FT_ULong*  length );

  /*
   * Used to implement FT_Get_Sfnt_Table().
   */
  typedef void*
  (*FT_SFNT_TableGetFunc)( FT_Face      face,
                           FT_Sfnt_Tag  tag );


  /*
   * Used to implement FT_Sfnt_Table_Info().
   */
  typedef FT_Error
  (*FT_SFNT_TableInfoFunc)( FT_Face    face,
                            FT_UInt    idx,
                            FT_ULong  *tag,
                            FT_ULong  *offset,
                            FT_ULong  *length );


  FT_DEFINE_SERVICE( SFNT_Table )
  {
    FT_SFNT_TableLoadFunc  load_table;
    FT_SFNT_TableGetFunc   get_table;
    FT_SFNT_TableInfoFunc  table_info;
  };


#define FT_DEFINE_SERVICE_SFNT_TABLEREC( class_, load_, get_, info_ )  \
  static const FT_Service_SFNT_TableRec  class_ =                      \
  {                                                                    \
    load_, get_, info_                                                 \
  };

  /* */


FT_END_HEADER


#endif /* SVSFNT_H_ */


/* END */
