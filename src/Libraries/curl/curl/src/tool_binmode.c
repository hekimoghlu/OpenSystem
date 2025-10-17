/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "tool_setup.h"

#ifdef HAVE_SETMODE

#ifdef HAVE_IO_H
#  include <io.h>
#endif

#ifdef HAVE_FCNTL_H
#  include <fcntl.h>
#endif

#include "tool_binmode.h"

#include "memdebug.h" /* keep this as LAST include */

void set_binmode(FILE *stream)
{
#ifdef O_BINARY
#  ifdef __HIGHC__
  _setmode(stream, O_BINARY);
#  else
  (void)setmode(fileno(stream), O_BINARY);
#  endif
#else
  (void)stream;
#endif
}

#endif /* HAVE_SETMODE */
