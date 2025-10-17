/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
/* NOTE: making 'symbols.chk' checks both that the exported
 * symbols in the library don't change and (implicitly) that
 * scripts/pnglibconf.h.prebuilt is as expected.
 * If scripts/pnglibconf.h.prebuilt is remade using
 * scripts/pnglibconf.dfa then this checks the .dfa file too.
 */

#define PNG_EXPORTA(ordinal, type, name, args, attributes)\
        PNG_DFN "@" name "@ @@" ordinal "@"
#define PNG_REMOVED(ordinal, type, name, args, attributes)\
        PNG_DFN "; @" name "@ @@" ordinal "@"
#define PNG_EXPORT_LAST_ORDINAL(ordinal)\
        PNG_DFN "; @@" ordinal "@"

/* Read the defaults, but use scripts/pnglibconf.h.prebuilt; the 'standard'
 * header file.
 */
#include "pnglibconf.h.prebuilt"
#include "../png.h"

/* Some things are turned off by default.  Turn these things
 * on here (by hand) to get the APIs they expose and validate
 * that no harm is done.  This list is the set of options
 * defaulted to 'off' in scripts/pnglibconf.dfa
 *
 * Maintenance: if scripts/pnglibconf.dfa options are changed
 * from, or to, 'disabled' this needs updating!
 */
#define PNG_BENIGN_ERRORS_SUPPORTED
#define PNG_ERROR_NUMBERS_SUPPORTED
#define PNG_READ_BIG_ENDIAN_SUPPORTED  /* should do nothing! */
#define PNG_INCH_CONVERSIONS_SUPPORTED
#define PNG_READ_16_TO_8_ACCURATE_SCALE_SUPPORTED
#define PNG_SET_OPTION_SUPPORTED

#undef PNG_H
#include "../png.h"

/* Finally there are a couple of places where option support
 * actually changes the APIs revealed using a #if/#else/#endif
 * test in png.h, test these here.
 */
#undef  PNG_FLOATING_POINT_SUPPORTED /* Exposes 'fixed' APIs */
#undef  PNG_ERROR_TEXT_SUPPORTED     /* Exposes unsupported APIs */

#undef PNG_H
#include "../png.h"
