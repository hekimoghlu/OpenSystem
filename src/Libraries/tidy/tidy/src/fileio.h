/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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

#ifndef __FILEIO_H__
#define __FILEIO_H__

/** @file fileio.h - does standard C I/O

  Implementation of a FILE* based TidyInputSource and 
  TidyOutputSink.

  (c) 1998-2006 (W3C) MIT, ERCIM, Keio University
  See tidy.h for the copyright notice.

  CVS Info:
    $Author$ 
    $Date$ 
    $Revision$ 
*/

#include "buffio.h"
#ifdef __cplusplus
extern "C" {
#endif

/** Allocate and initialize file input source */
int TY_(initFileSource)( TidyInputSource* source, FILE* fp );

/** Free file input source */
void TY_(freeFileSource)( TidyInputSource* source, Bool closeIt );

/** Initialize file output sink */
void TY_(initFileSink)( TidyOutputSink* sink, FILE* fp );

/* Needed for internal declarations */
void TIDY_CALL TY_(filesink_putByte)( void* sinkData, byte bv );

#ifdef __cplusplus
}
#endif
#endif /* __FILEIO_H__ */
