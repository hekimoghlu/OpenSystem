/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#ifndef FTGZIP_H_
#define FTGZIP_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER

  /**************************************************************************
   *
   * @section:
   *   gzip
   *
   * @title:
   *   GZIP Streams
   *
   * @abstract:
   *   Using gzip-compressed font files.
   *
   * @description:
   *   This section contains the declaration of Gzip-specific functions.
   *
   */


  /**************************************************************************
   *
   * @function:
   *   FT_Stream_OpenGzip
   *
   * @description:
   *   Open a new stream to parse gzip-compressed font files.  This is mainly
   *   used to support the compressed `*.pcf.gz` fonts that come with
   *   XFree86.
   *
   * @input:
   *   stream ::
   *     The target embedding stream.
   *
   *   source ::
   *     The source stream.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   The source stream must be opened _before_ calling this function.
   *
   *   Calling the internal function `FT_Stream_Close` on the new stream will
   *   **not** call `FT_Stream_Close` on the source stream.  None of the
   *   stream objects will be released to the heap.
   *
   *   The stream implementation is very basic and resets the decompression
   *   process each time seeking backwards is needed within the stream.
   *
   *   In certain builds of the library, gzip compression recognition is
   *   automatically handled when calling @FT_New_Face or @FT_Open_Face.
   *   This means that if no font driver is capable of handling the raw
   *   compressed file, the library will try to open a gzipped stream from it
   *   and re-open the face with it.
   *
   *   This function may return `FT_Err_Unimplemented_Feature` if your build
   *   of FreeType was not compiled with zlib support.
   */
  FT_EXPORT( FT_Error )
  FT_Stream_OpenGzip( FT_Stream  stream,
                      FT_Stream  source );


  /**************************************************************************
   *
   * @function:
   *   FT_Gzip_Uncompress
   *
   * @description:
   *   Decompress a zipped input buffer into an output buffer.  This function
   *   is modeled after zlib's `uncompress` function.
   *
   * @input:
   *   memory ::
   *     A FreeType memory handle.
   *
   *   input ::
   *     The input buffer.
   *
   *   input_len ::
   *     The length of the input buffer.
   *
   * @output:
   *   output ::
   *     The output buffer.
   *
   * @inout:
   *   output_len ::
   *     Before calling the function, this is the total size of the output
   *     buffer, which must be large enough to hold the entire uncompressed
   *     data (so the size of the uncompressed data must be known in
   *     advance).  After calling the function, `output_len` is the size of
   *     the used data in `output`.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This function may return `FT_Err_Unimplemented_Feature` if your build
   *   of FreeType was not compiled with zlib support.
   *
   * @since:
   *   2.5.1
   */
  FT_EXPORT( FT_Error )
  FT_Gzip_Uncompress( FT_Memory       memory,
                      FT_Byte*        output,
                      FT_ULong*       output_len,
                      const FT_Byte*  input,
                      FT_ULong        input_len );

  /* */


FT_END_HEADER

#endif /* FTGZIP_H_ */


/* END */
