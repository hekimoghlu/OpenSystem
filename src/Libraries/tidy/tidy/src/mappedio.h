/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#ifndef __TIDY_MAPPED_IO_H__
#define __TIDY_MAPPED_IO_H__

/* Interface to mmap style I/O

   (c) 2006 (W3C) MIT, ERCIM, Keio University
   See tidy.h for the copyright notice.

   $Id$
*/

#if defined(_WIN32)
int TY_(DocParseFileWithMappedFile)( TidyDocImpl* doc, ctmbstr filnam );
#endif

#endif /* __TIDY_MAPPED_IO_H__ */
