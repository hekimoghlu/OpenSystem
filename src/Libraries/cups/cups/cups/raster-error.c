/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#include "cups-private.h"
#include "raster-private.h"
#include "debug-internal.h"


/*
 * '_cupsRasterAddError()' - Add an error message to the error buffer.
 */

void
_cupsRasterAddError(const char *f,	/* I - Printf-style error message */
                    ...)		/* I - Additional arguments as needed */
{
  _cups_globals_t	*cg = _cupsGlobals();
					/* Thread globals */
  _cups_raster_error_t	*buf = &cg->raster_error;
					/* Error buffer */
  va_list	ap;			/* Pointer to additional arguments */
  char		s[2048];		/* Message string */
  ssize_t	bytes;			/* Bytes in message string */


  DEBUG_printf(("_cupsRasterAddError(f=\"%s\", ...)", f));

  va_start(ap, f);
  bytes = vsnprintf(s, sizeof(s), f, ap);
  va_end(ap);

  if (bytes <= 0)
    return;

  DEBUG_printf(("1_cupsRasterAddError: %s", s));

  bytes ++;

  if ((size_t)bytes >= sizeof(s))
    return;

  if (bytes > (ssize_t)(buf->end - buf->current))
  {
   /*
    * Allocate more memory...
    */

    char	*temp;			/* New buffer */
    size_t	size;			/* Size of buffer */


    size = (size_t)(buf->end - buf->start + 2 * bytes + 1024);

    if (buf->start)
      temp = realloc(buf->start, size);
    else
      temp = malloc(size);

    if (!temp)
      return;

   /*
    * Update pointers...
    */

    buf->end     = temp + size;
    buf->current = temp + (buf->current - buf->start);
    buf->start   = temp;
  }

 /*
  * Append the message to the end of the current string...
  */

  memcpy(buf->current, s, (size_t)bytes);
  buf->current += bytes - 1;
}


/*
 * '_cupsRasterClearError()' - Clear the error buffer.
 */

void
_cupsRasterClearError(void)
{
  _cups_globals_t	*cg = _cupsGlobals();
					/* Thread globals */
  _cups_raster_error_t	*buf = &cg->raster_error;
					/* Error buffer */


  buf->current = buf->start;

  if (buf->start)
    *(buf->start) = '\0';
}


/*
 * '_cupsRasterErrorString()' - Return the last error from a raster function.
 *
 * If there are no recent errors, NULL is returned.
 *
 * @since CUPS 1.3/macOS 10.5@
 */

const char *				/* O - Last error */
_cupsRasterErrorString(void)
{
  _cups_globals_t	*cg = _cupsGlobals();
					/* Thread globals */
  _cups_raster_error_t	*buf = &cg->raster_error;
					/* Error buffer */


  if (buf->current == buf->start)
    return (NULL);
  else
    return (buf->start);
}
