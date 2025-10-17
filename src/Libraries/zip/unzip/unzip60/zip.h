/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
/* This is a dummy zip.h to allow the source files shared with Zip
   (crypt.c, crc32.c, ttyio.c, win32/win32i64.c) to compile for UnZip.
   (In case you are looking for the Info-ZIP license, please follow
   the pointers above.)  */

#ifndef __zip_h   /* don't include more than once */
#define __zip_h

#define UNZIP_INTERNAL
#include "unzip.h"

#define local static

#define ZE_MEM         PK_MEM
#define ziperr(c, h)   return

#endif /* !__zip_h */
