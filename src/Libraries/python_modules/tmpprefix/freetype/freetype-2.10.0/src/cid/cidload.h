/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#ifndef CIDLOAD_H_
#define CIDLOAD_H_


#include <ft2build.h>
#include FT_INTERNAL_STREAM_H
#include "cidparse.h"


FT_BEGIN_HEADER


  typedef struct  CID_Loader_
  {
    CID_Parser  parser;          /* parser used to read the stream */
    FT_Int      num_chars;       /* number of characters in encoding */

  } CID_Loader;


  FT_LOCAL( FT_ULong )
  cid_get_offset( FT_Byte**  start,
                  FT_Byte    offsize );

  FT_LOCAL( FT_Error )
  cid_face_open( CID_Face  face,
                 FT_Int    face_index );


FT_END_HEADER

#endif /* CIDLOAD_H_ */


/* END */
