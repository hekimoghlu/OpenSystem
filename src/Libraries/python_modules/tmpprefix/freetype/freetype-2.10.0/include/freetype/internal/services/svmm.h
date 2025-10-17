/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#ifndef SVMM_H_
#define SVMM_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   * A service used to manage multiple-masters data in a given face.
   *
   * See the related APIs in `ftmm.h' (FT_MULTIPLE_MASTERS_H).
   *
   */

#define FT_SERVICE_ID_MULTI_MASTERS  "multi-masters"


  typedef FT_Error
  (*FT_Get_MM_Func)( FT_Face           face,
                     FT_Multi_Master*  master );

  typedef FT_Error
  (*FT_Get_MM_Var_Func)( FT_Face      face,
                         FT_MM_Var*  *master );

  typedef FT_Error
  (*FT_Set_MM_Design_Func)( FT_Face   face,
                            FT_UInt   num_coords,
                            FT_Long*  coords );

  /* use return value -1 to indicate that the new coordinates  */
  /* are equal to the current ones; no changes are thus needed */
  typedef FT_Error
  (*FT_Set_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  /* use return value -1 to indicate that the new coordinates  */
  /* are equal to the current ones; no changes are thus needed */
  typedef FT_Error
  (*FT_Set_MM_Blend_Func)( FT_Face   face,
                           FT_UInt   num_coords,
                           FT_Long*  coords );

  typedef FT_Error
  (*FT_Get_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Set_Instance_Func)( FT_Face  face,
                           FT_UInt  instance_index );

  typedef FT_Error
  (*FT_Get_MM_Blend_Func)( FT_Face   face,
                           FT_UInt   num_coords,
                           FT_Long*  coords );

  typedef FT_Error
  (*FT_Get_Var_Blend_Func)( FT_Face      face,
                            FT_UInt     *num_coords,
                            FT_Fixed*   *coords,
                            FT_Fixed*   *normalizedcoords,
                            FT_MM_Var*  *mm_var );

  typedef void
  (*FT_Done_Blend_Func)( FT_Face );

  typedef FT_Error
  (*FT_Set_MM_WeightVector_Func)( FT_Face    face,
                                  FT_UInt    len,
                                  FT_Fixed*  weight_vector );

  typedef FT_Error
  (*FT_Get_MM_WeightVector_Func)( FT_Face    face,
                                  FT_UInt*   len,
                                  FT_Fixed*  weight_vector );


  FT_DEFINE_SERVICE( MultiMasters )
  {
    FT_Get_MM_Func               get_mm;
    FT_Set_MM_Design_Func        set_mm_design;
    FT_Set_MM_Blend_Func         set_mm_blend;
    FT_Get_MM_Blend_Func         get_mm_blend;
    FT_Get_MM_Var_Func           get_mm_var;
    FT_Set_Var_Design_Func       set_var_design;
    FT_Get_Var_Design_Func       get_var_design;
    FT_Set_Instance_Func         set_instance;
    FT_Set_MM_WeightVector_Func  set_mm_weightvector;
    FT_Get_MM_WeightVector_Func  get_mm_weightvector;

    /* for internal use; only needed for code sharing between modules */
    FT_Get_Var_Blend_Func  get_var_blend;
    FT_Done_Blend_Func     done_blend;
  };


#define FT_DEFINE_SERVICE_MULTIMASTERSREC( class_,            \
                                           get_mm_,           \
                                           set_mm_design_,    \
                                           set_mm_blend_,     \
                                           get_mm_blend_,     \
                                           get_mm_var_,       \
                                           set_var_design_,   \
                                           get_var_design_,   \
                                           set_instance_,     \
                                           set_weightvector_, \
                                           get_weightvector_, \
                                           get_var_blend_,    \
                                           done_blend_ )      \
  static const FT_Service_MultiMastersRec  class_ =           \
  {                                                           \
    get_mm_,                                                  \
    set_mm_design_,                                           \
    set_mm_blend_,                                            \
    get_mm_blend_,                                            \
    get_mm_var_,                                              \
    set_var_design_,                                          \
    get_var_design_,                                          \
    set_instance_,                                            \
    set_weightvector_,                                        \
    get_weightvector_,                                        \
    get_var_blend_,                                           \
    done_blend_                                               \
  };

  /* */


FT_END_HEADER

#endif /* SVMM_H_ */


/* END */
