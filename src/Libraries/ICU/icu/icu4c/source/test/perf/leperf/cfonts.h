/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#ifndef __CFONTS_H
#define __CFONTS_H

#include "LETypes.h"
#include "loengine.h"

le_font *le_portableFontOpen(const char *fileName,
							float pointSize,
							LEErrorCode *status);

le_font *le_simpleFontOpen(float pointSize,
						   LEErrorCode *status);

void le_fontClose(le_font *font);

const char *le_getNameString(le_font *font, le_uint16 nameID, le_uint16 platform, le_uint16 encoding, le_uint16 language);

const LEUnicode16 *le_getUnicodeNameString(le_font *font, le_uint16 nameID, le_uint16 platform, le_uint16 encoding, le_uint16 language);

void le_deleteNameString(le_font *font, const char *name);

void le_deleteUnicodeNameString(le_font *font, const LEUnicode16 *name);

le_uint32 le_getFontChecksum(le_font *font);

#endif
