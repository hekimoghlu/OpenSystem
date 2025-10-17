/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
// Test program for message catalog class.
//
// Copyright Â©Â 2008-2019 by Apple Inc.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more
// information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"


//
// 'main()' - Open a message catalog
//

int					// O - Exit status
main(int  argc,				// I - Number of command-line arguments
     char *argv[])			// I - Command-line arguments
{
  ppdcCatalog	*catalog;		// Message catalog
  ppdcMessage	*m;			// Current message


  if (argc != 2)
  {
    puts("Usage: testcatalog filename");
    return (1);
  }

  // Scan the command-line...
  catalog = new ppdcCatalog(NULL, argv[1]);

  printf("%s: %u messages\n", argv[1], (unsigned)catalog->messages->count);

  for (m = (ppdcMessage *)catalog->messages->first();
       m;
       m = (ppdcMessage *)catalog->messages->next())
    printf("%s: %s\n", m->id->value, m->string->value);

  catalog->release();

  // Return with no errors.
  return (0);
}
