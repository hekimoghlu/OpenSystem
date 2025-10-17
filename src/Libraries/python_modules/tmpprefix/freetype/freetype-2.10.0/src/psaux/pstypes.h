/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#ifndef PSTYPES_H_
#define PSTYPES_H_

#include <ft2build.h>
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  /*
   * The data models that we expect to support are as follows:
   *
   *   name  char short int long long-long pointer example
   *  -----------------------------------------------------
   *   ILP32  8    16    32  32     64*      32    32-bit MacOS, x86
   *   LLP64  8    16    32  32     64       64    x64
   *   LP64   8    16    32  64     64       64    64-bit MacOS
   *
   *    *) type may be supported by emulation on a 32-bit architecture
   *
   */


  /* integers at least 32 bits wide */
#define CF2_UInt  FT_UFast
#define CF2_Int   FT_Fast


  /* fixed-float numbers */
  typedef FT_Int32  CF2_F16Dot16;


FT_END_HEADER


#endif /* PSTYPES_H_ */


/* END */
