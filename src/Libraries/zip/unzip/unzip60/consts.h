/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
/*---------------------------------------------------------------------------

  consts.h

  This file contains global, initialized variables that never change.  It is
  included by unzip.c and windll/windll.c.

  ---------------------------------------------------------------------------*/


/* And'ing with mask_bits[n] masks the lower n bits */
ZCONST unsigned near mask_bits[17] = {
    0x0000,
    0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
    0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff
};

ZCONST char Far VersionDate[] = UZ_VERSION_DATE; /* now defined in unzvers.h */

#ifndef SFX
   ZCONST char Far EndSigMsg[] =
   "\nnote:  didn't find end-of-central-dir signature at end of central dir.\n";
#endif

ZCONST char Far CentSigMsg[] =
  "error:  expected central file header signature not found (file #%lu).\n";
ZCONST char Far SeekMsg[] =
  "error [%s]:  attempt to seek before beginning of zipfile\n%s";
ZCONST char Far FilenameNotMatched[] = "caution: filename not matched:  %s\n";
ZCONST char Far ExclFilenameNotMatched[] =
  "caution: excluded filename not matched:  %s\n";

#ifdef VMS
  ZCONST char Far ReportMsg[] = "\
  (please check that you have transferred or created the zipfile in the\n\
  appropriate BINARY mode--this includes ftp, Kermit, AND unzip'd zipfiles)\n";
#else
  ZCONST char Far ReportMsg[] = "\
  (please check that you have transferred or created the zipfile in the\n\
  appropriate BINARY mode and that you have compiled UnZip properly)\n";
#endif

#ifndef SFX
  ZCONST char Far Zipnfo[] = "zipinfo";
  ZCONST char Far CompiledWith[] = "Compiled with %s%s for %s%s%s%s.\n\n";
#endif
