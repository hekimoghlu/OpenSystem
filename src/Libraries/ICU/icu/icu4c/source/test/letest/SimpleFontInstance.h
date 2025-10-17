/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
 *******************************************************************************
 *
 *   Copyright (C) 1999-2015, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *
 *******************************************************************************
 *   file name:  SimpleFontInstance.h
 *
 *   created on: 03/30/2006
 *   created by: Eric R. Mader
 */

#ifndef __SIMPLEFONTINSTANCE_H
#define __SIMPLEFONTINSTANCE_H

#include "layout/LETypes.h"
#include "layout/LEFontInstance.h"

U_NAMESPACE_USE

class SimpleFontInstance : public LEFontInstance
{
private:
    float     fPointSize;
    le_int32  fAscent;
    le_int32  fDescent;

protected:
    const void *readFontTable(LETag tableTag) const;

public:
    SimpleFontInstance(float pointSize, LEErrorCode &status);

    virtual ~SimpleFontInstance();

    const void *getFontTable(LETag tableTag, size_t &length) const override;

    le_int32 getUnitsPerEM() const override;

    le_int32 getAscent() const override;

    le_int32 getDescent() const override;

    le_int32 getLeading() const override;

    // We really want to inherit this method from the superclass, but some compilers
    // issue a warning if we don't implement it...
    LEGlyphID mapCharToGlyph(LEUnicode32 ch, const LECharMapper *mapper, le_bool filterZeroWidth) const override;
    
    // We really want to inherit this method from the superclass, but some compilers
    // issue a warning if we don't implement it...
    LEGlyphID mapCharToGlyph(LEUnicode32 ch, const LECharMapper *mapper) const override;

    LEGlyphID mapCharToGlyph(LEUnicode32 ch) const override;

    void getGlyphAdvance(LEGlyphID glyph, LEPoint &advance) const override;

    le_bool getGlyphPoint(LEGlyphID glyph, le_int32 pointNumber, LEPoint &point) const override;

    float getXPixelsPerEm() const override;

    float getYPixelsPerEm() const override;

    float getScaleFactorX() const override;

    float getScaleFactorY() const override;

};

#endif
