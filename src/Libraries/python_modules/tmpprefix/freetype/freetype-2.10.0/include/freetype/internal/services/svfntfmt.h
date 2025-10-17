/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#ifndef SVFNTFMT_H_
#define SVFNTFMT_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   * A trivial service used to return the name of a face's font driver,
   * according to the XFree86 nomenclature.  Note that the service data is a
   * simple constant string pointer.
   */

#define FT_SERVICE_ID_FONT_FORMAT  "font-format"

#define FT_FONT_FORMAT_TRUETYPE  "TrueType"
#define FT_FONT_FORMAT_TYPE_1    "Type 1"
#define FT_FONT_FORMAT_BDF       "BDF"
#define FT_FONT_FORMAT_PCF       "PCF"
#define FT_FONT_FORMAT_TYPE_42   "Type 42"
#define FT_FONT_FORMAT_CID       "CID Type 1"
#define FT_FONT_FORMAT_CFF       "CFF"
#define FT_FONT_FORMAT_PFR       "PFR"
#define FT_FONT_FORMAT_WINFNT    "Windows FNT"

  /* */


FT_END_HEADER


#endif /* SVFNTFMT_H_ */


/* END */
