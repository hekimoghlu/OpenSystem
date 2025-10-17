/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#include "config.h"

#include <pal/spi/cf/CoreTextSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, CoreText, PAL_EXPORT)

// FIXME: Move this to strong linking as soon as people have a chance to update to an SDK that includes it.
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(PAL, CoreText, CTFontCopyColorGlyphCoverage, CFBitVectorRef, (CTFontRef font), (font))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, CoreText, CTFontManagerCreateMemorySafeFontDescriptorFromData, CTFontDescriptorRef, (CFDataRef data), (data), PAL_EXPORT)

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE(PAL, OTSVG)

SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OTSVG, OTSVGTableCreateFromData, OTSVGTableRef, (CFDataRef svgTable, unsigned unitsPerEm, CGFloat fontSize), (svgTable, unitsPerEm, fontSize))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OTSVG, OTSVGTableGetDocumentIndexForGlyph, CFIndex, (OTSVGTableRef table, CGGlyph glyph), (table, glyph))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, OTSVG, OTSVGTableRelease, void, (OTSVGTableRef table), (table))
