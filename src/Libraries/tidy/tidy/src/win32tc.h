/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

#ifndef __WIN32TC_H__
#define __WIN32TC_H__
#ifdef TIDY_WIN32_MLANG_SUPPORT

/* win32tc.h -- Interface to Win32 transcoding routines

   (c) 1998-2006 (W3C) MIT, ERCIM, Keio University
   See tidy.h for the copyright notice.

   $Id$
*/

uint TY_(Win32MLangGetCPFromName)(ctmbstr encoding);
Bool TY_(Win32MLangInitInputTranscoder)(StreamIn * in, uint wincp);
void TY_(Win32MLangUninitInputTranscoder)(StreamIn * in);
int TY_(Win32MLangGetChar)(byte firstByte, StreamIn * in, uint * bytesRead);

#endif /* TIDY_WIN32_MLANG_SUPPORT */
#endif /* __WIN32TC_H__ */
