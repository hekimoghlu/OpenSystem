/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#ifndef PSREAD_H_
#define PSREAD_H_


FT_BEGIN_HEADER


  typedef struct  CF2_BufferRec_
  {
    FT_Error*       error;
    const FT_Byte*  start;
    const FT_Byte*  end;
    const FT_Byte*  ptr;

  } CF2_BufferRec, *CF2_Buffer;


  FT_LOCAL( CF2_Int )
  cf2_buf_readByte( CF2_Buffer  buf );
  FT_LOCAL( FT_Bool )
  cf2_buf_isEnd( CF2_Buffer  buf );


FT_END_HEADER


#endif /* PSREAD_H_ */


/* END */
