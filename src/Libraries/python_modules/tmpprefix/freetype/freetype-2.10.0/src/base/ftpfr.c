/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include FT_INTERNAL_DEBUG_H

#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_PFR_H


  /* check the format */
  static FT_Service_PfrMetrics
  ft_pfr_check( FT_Face  face )
  {
    FT_Service_PfrMetrics  service = NULL;


    if ( face )
      FT_FACE_LOOKUP_SERVICE( face, service, PFR_METRICS );

    return service;
  }


  /* documentation is in ftpfr.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PFR_Metrics( FT_Face    face,
                      FT_UInt   *aoutline_resolution,
                      FT_UInt   *ametrics_resolution,
                      FT_Fixed  *ametrics_x_scale,
                      FT_Fixed  *ametrics_y_scale )
  {
    FT_Error               error = FT_Err_Ok;
    FT_Service_PfrMetrics  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    service = ft_pfr_check( face );
    if ( service )
    {
      error = service->get_metrics( face,
                                    aoutline_resolution,
                                    ametrics_resolution,
                                    ametrics_x_scale,
                                    ametrics_y_scale );
    }
    else
    {
      FT_Fixed  x_scale, y_scale;


      /* this is not a PFR font */
      if ( aoutline_resolution )
        *aoutline_resolution = face->units_per_EM;

      if ( ametrics_resolution )
        *ametrics_resolution = face->units_per_EM;

      x_scale = y_scale = 0x10000L;
      if ( face->size )
      {
        x_scale = face->size->metrics.x_scale;
        y_scale = face->size->metrics.y_scale;
      }

      if ( ametrics_x_scale )
        *ametrics_x_scale = x_scale;

      if ( ametrics_y_scale )
        *ametrics_y_scale = y_scale;

      error = FT_THROW( Unknown_File_Format );
    }

    return error;
  }


  /* documentation is in ftpfr.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PFR_Kerning( FT_Face     face,
                      FT_UInt     left,
                      FT_UInt     right,
                      FT_Vector  *avector )
  {
    FT_Error               error;
    FT_Service_PfrMetrics  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !avector )
      return FT_THROW( Invalid_Argument );

    service = ft_pfr_check( face );
    if ( service )
      error = service->get_kerning( face, left, right, avector );
    else
      error = FT_Get_Kerning( face, left, right,
                              FT_KERNING_UNSCALED, avector );

    return error;
  }


  /* documentation is in ftpfr.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PFR_Advance( FT_Face   face,
                      FT_UInt   gindex,
                      FT_Pos   *aadvance )
  {
    FT_Error               error;
    FT_Service_PfrMetrics  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !aadvance )
      return FT_THROW( Invalid_Argument );

    service = ft_pfr_check( face );
    if ( service )
      error = service->get_advance( face, gindex, aadvance );
    else
      /* XXX: TODO: PROVIDE ADVANCE-LOADING METHOD TO ALL FONT DRIVERS */
      error = FT_THROW( Invalid_Argument );

    return error;
  }


/* END */
