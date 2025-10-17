/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#ifndef __CMAPS_H
#define __CMAPS_H

#include "layout/LETypes.h"
//#include "letest.h"
#include "sfnt.h"

class CMAPMapper
{
public:
    virtual LEGlyphID unicodeToGlyph(LEUnicode32 unicode32) const = 0;

    virtual ~CMAPMapper();

    static CMAPMapper *createUnicodeMapper(const CMAPTable *cmap);

protected:
    CMAPMapper(const CMAPTable *cmap);

    CMAPMapper() {}

private:
    const CMAPTable *fcmap;
};

class CMAPFormat4Mapper : public CMAPMapper
{
public:
    CMAPFormat4Mapper(const CMAPTable *cmap, const CMAPFormat4Encoding *header);

    virtual ~CMAPFormat4Mapper();

    LEGlyphID unicodeToGlyph(LEUnicode32 unicode32) const override;

protected:
    CMAPFormat4Mapper() {}

private:
    le_uint16       fEntrySelector;
    le_uint16       fRangeShift;
    const le_uint16 *fEndCodes;
    const le_uint16 *fStartCodes;
    const le_uint16 *fIdDelta;
    const le_uint16 *fIdRangeOffset;
};

class CMAPGroupMapper : public CMAPMapper
{
public:
    CMAPGroupMapper(const CMAPTable *cmap, const CMAPGroup *groups, le_uint32 nGroups);

    virtual ~CMAPGroupMapper();

    LEGlyphID unicodeToGlyph(LEUnicode32 unicode32) const override;

protected:
    CMAPGroupMapper() {}

private:
    le_int32 fPower;
    le_int32 fRangeOffset;
    const CMAPGroup *fGroups;
};

inline CMAPMapper::CMAPMapper(const CMAPTable *cmap)
    : fcmap(cmap)
{
    // nothing else to do
}

inline CMAPMapper::~CMAPMapper()
{
    LE_DELETE_ARRAY(fcmap);
}

#endif

