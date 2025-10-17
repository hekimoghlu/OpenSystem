/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
/*
 * Include necessary headers...
 */

#include "cgi.h"


/*
 * 'main()' - Test the CGI code.
 */

int					/* O - Exit status */
main(void)
{
 /*
  * Test file upload/multi-part submissions...
  */

  freopen("multipart.dat", "rb", stdin);

  putenv("CONTENT_TYPE=multipart/form-data; "
         "boundary=---------------------------1977426492562745908748943111");
  putenv("REQUEST_METHOD=POST");

  printf("cgiInitialize: ");
  if (cgiInitialize())
  {
    const cgi_file_t	*file;		/* Upload file */

    if ((file = cgiGetFile()) != NULL)
    {
      puts("PASS");
      printf("    tempfile=\"%s\"\n", file->tempfile);
      printf("    name=\"%s\"\n", file->name);
      printf("    filename=\"%s\"\n", file->filename);
      printf("    mimetype=\"%s\"\n", file->mimetype);
    }
    else
      puts("FAIL (no file!)");
  }
  else
    puts("FAIL (init)");

 /*
  * Return with no errors...
  */

  return (0);
}
