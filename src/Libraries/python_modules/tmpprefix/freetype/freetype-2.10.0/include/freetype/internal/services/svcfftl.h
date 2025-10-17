/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#ifndef SVCFFTL_H_
#define SVCFFTL_H_

#include FT_INTERNAL_SERVICE_H
#include FT_INTERNAL_CFF_TYPES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_CFF_LOAD  "cff-load"


  typedef FT_UShort
  (*FT_Get_Standard_Encoding_Func)( FT_UInt  charcode );

  typedef FT_Error
  (*FT_Load_Private_Dict_Func)( CFF_Font     font,
                                CFF_SubFont  subfont,
                                FT_UInt      lenNDV,
                                FT_Fixed*    NDV );

  typedef FT_Byte
  (*FT_FD_Select_Get_Func)( CFF_FDSelect  fdselect,
                            FT_UInt       glyph_index );

  typedef FT_Bool
  (*FT_Blend_Check_Vector_Func)( CFF_Blend  blend,
                                 FT_UInt    vsindex,
                                 FT_UInt    lenNDV,
                                 FT_Fixed*  NDV );

  typedef FT_Error
  (*FT_Blend_Build_Vector_Func)( CFF_Blend  blend,
                                 FT_UInt    vsindex,
                                 FT_UInt    lenNDV,
                                 FT_Fixed*  NDV );


  FT_DEFINE_SERVICE( CFFLoad )
  {
    FT_Get_Standard_Encoding_Func  get_standard_encoding;
    FT_Load_Private_Dict_Func      load_private_dict;
    FT_FD_Select_Get_Func          fd_select_get;
    FT_Blend_Check_Vector_Func     blend_check_vector;
    FT_Blend_Build_Vector_Func     blend_build_vector;
  };


#define FT_DEFINE_SERVICE_CFFLOADREC( class_,                  \
                                      get_standard_encoding_,  \
                                      load_private_dict_,      \
                                      fd_select_get_,          \
                                      blend_check_vector_,     \
                                      blend_build_vector_ )    \
  static const FT_Service_CFFLoadRec  class_ =                 \
  {                                                            \
    get_standard_encoding_,                                    \
    load_private_dict_,                                        \
    fd_select_get_,                                            \
    blend_check_vector_,                                       \
    blend_build_vector_                                        \
  };


FT_END_HEADER


#endif


/* END */
