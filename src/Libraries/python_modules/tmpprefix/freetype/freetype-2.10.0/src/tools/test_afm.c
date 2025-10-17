/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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
#include FT_FREETYPE_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H

  void dump_fontinfo( AFM_FontInfo  fi )
  {
    FT_UInt  i;


    printf( "This AFM is for %sCID font.\n\n",
            ( fi->IsCIDFont ) ? "" : "non-" );

    printf( "FontBBox: %.2f %.2f %.2f %.2f\n", fi->FontBBox.xMin / 65536.,
                                               fi->FontBBox.yMin / 65536.,
                                               fi->FontBBox.xMax / 65536.,
                                               fi->FontBBox.yMax / 65536. );
    printf( "Ascender: %.2f\n", fi->Ascender / 65536. );
    printf( "Descender: %.2f\n\n", fi->Descender / 65536. );

    if ( fi->NumTrackKern )
      printf( "There are %d sets of track kernings:\n",
              fi->NumTrackKern );
    else
      printf( "There is no track kerning.\n" );

    for ( i = 0; i < fi->NumTrackKern; i++ )
    {
      AFM_TrackKern  tk = fi->TrackKerns + i;


      printf( "\t%2d: %5.2f %5.2f %5.2f %5.2f\n", tk->degree,
                                                  tk->min_ptsize / 65536.,
                                                  tk->min_kern / 65536.,
                                                  tk->max_ptsize / 65536.,
                                                  tk->max_kern / 65536. );
    }

    printf( "\n" );

    if ( fi->NumKernPair )
      printf( "There are %d kerning pairs:\n",
              fi->NumKernPair );
    else
      printf( "There is no kerning pair.\n" );

    for ( i = 0; i < fi->NumKernPair; i++ )
    {
      AFM_KernPair  kp = fi->KernPairs + i;


      printf( "\t%3d + %3d => (%4d, %4d)\n", kp->index1,
                                             kp->index2,
                                             kp->x,
                                             kp->y );
    }

  }

  int
  dummy_get_index( const char*  name,
                   FT_Offset    len,
                   void*        user_data )
  {
    if ( len )
      return name[0];
    else
      return 0;
  }

  FT_Error
  parse_afm( FT_Library    library,
             FT_Stream     stream,
             AFM_FontInfo  fi )
  {
    PSAux_Service  psaux;
    AFM_ParserRec  parser;
    FT_Error       error = FT_Err_Ok;


    psaux = (PSAux_Service)FT_Get_Module_Interface( library, "psaux" );
    if ( !psaux || !psaux->afm_parser_funcs )
      return -1;

    error = FT_Stream_EnterFrame( stream, stream->size );
    if ( error )
      return error;

    error = psaux->afm_parser_funcs->init( &parser,
                                           library->memory,
                                           stream->cursor,
                                           stream->limit );
    if ( error )
      return error;

    parser.FontInfo = fi;
    parser.get_index = dummy_get_index;

    error = psaux->afm_parser_funcs->parse( &parser );

    psaux->afm_parser_funcs->done( &parser );

    return error;
  }


  int main( int    argc,
            char** argv )
  {
    FT_Library       library;
    FT_StreamRec     stream;
    FT_Error         error = FT_Err_Ok;
    AFM_FontInfoRec  fi;


    if ( argc < 2 )
      return FT_ERR( Invalid_Argument );

    error = FT_Init_FreeType( &library );
    if ( error )
      return error;

    FT_ZERO( &stream );
    error = FT_Stream_Open( &stream, argv[1] );
    if ( error )
      goto Exit;
    stream.memory = library->memory;

    FT_ZERO( &fi );
    error = parse_afm( library, &stream, &fi );

    if ( !error )
    {
      FT_Memory  memory = library->memory;


      dump_fontinfo( &fi );

      if ( fi.KernPairs )
        FT_FREE( fi.KernPairs );
      if ( fi.TrackKerns )
        FT_FREE( fi.TrackKerns );
    }
    else
      printf( "parse error\n" );

    FT_Stream_Close( &stream );

  Exit:
    FT_Done_FreeType( library );

    return error;
  }
