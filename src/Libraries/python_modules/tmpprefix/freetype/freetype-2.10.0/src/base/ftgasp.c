/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include FT_GASP_H
#include FT_INTERNAL_TRUETYPE_TYPES_H


  FT_EXPORT_DEF( FT_Int )
  FT_Get_Gasp( FT_Face  face,
               FT_UInt  ppem )
  {
    FT_Int  result = FT_GASP_NO_TABLE;


    if ( face && FT_IS_SFNT( face ) )
    {
      TT_Face  ttface = (TT_Face)face;


      if ( ttface->gasp.numRanges > 0 )
      {
        TT_GaspRange  range     = ttface->gasp.gaspRanges;
        TT_GaspRange  range_end = range + ttface->gasp.numRanges;


        while ( ppem > range->maxPPEM )
        {
          range++;
          if ( range >= range_end )
            goto Exit;
        }

        result = range->gaspFlag;

        /* ensure that we don't have spurious bits */
        if ( ttface->gasp.version == 0 )
          result &= 3;
      }
    }
  Exit:
    return result;
  }


/* END */
