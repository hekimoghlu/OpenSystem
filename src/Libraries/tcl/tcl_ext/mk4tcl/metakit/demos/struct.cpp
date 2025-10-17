/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

//  This command-line utility displays the data structure of a datafile
//  created with the Metakit library as a one-line description.

#include "mk4.h"

#include <stdio.h>

#if defined (macintosh)
#include /**/ <console.h>
#define d4_InitMain(c,v)  c = ccommand(&v)
#endif

/////////////////////////////////////////////////////////////////////////////
  
int main(int argc, char** argv)
{
#ifdef d4_InitMain
  d4_InitMain(argc, argv);
#endif

  if (argc != 2)
    fputs("Usage: STRUCT datafile", stderr);
  else
  {
    c4_Storage store (argv[1], false);
    puts(store.Description());
  }
    
  return 0;
}

/////////////////////////////////////////////////////////////////////////////
