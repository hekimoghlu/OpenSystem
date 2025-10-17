/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

//
// File class for the CUPS PPD Compiler.
//
// Copyright 2007-2010 by Apple Inc.
// Copyright 2002-2005 by Easy Software Products.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"


//
// 'ppdcFile::ppdcFile()' - Create (open) a file.
//

ppdcFile::ppdcFile(const char  *f,		// I - File to open
                   cups_file_t *ffp)		// I - File pointer to use
{
  if (ffp)
  {
    fp = ffp;
    cupsFileRewind(fp);
  }
  else
    fp = cupsFileOpen(f, "r");

  close_on_delete = !ffp;
  filename        = f;
  line            = 1;

  if (!fp)
    _cupsLangPrintf(stderr, _("ppdc: Unable to open %s: %s"), f,
                    strerror(errno));
}


//
// 'ppdcFile::~ppdcFile()' - Delete (close) a file.
//

ppdcFile::~ppdcFile()
{
  if (close_on_delete && fp)
    cupsFileClose(fp);
}


//
// 'ppdcFile::get()' - Get a character from a file.
//

int
ppdcFile::get()
{
  int	ch;					// Character from file


  // Return EOF if there is no open file...
  if (!fp)
    return (EOF);

  // Get the character...
  ch = cupsFileGetChar(fp);

  // Update the line number as needed...
  if (ch == '\n')
    line ++;

  // Return the character...
  return (ch);
}


//
// 'ppdcFile::peek()' - Look at the next character from a file.
//

int					// O - Next character in file
ppdcFile::peek()
{
  // Return immediaely if there is no open file...
  if (!fp)
    return (EOF);

  // Otherwise return the next character without advancing...
  return (cupsFilePeekChar(fp));
}
