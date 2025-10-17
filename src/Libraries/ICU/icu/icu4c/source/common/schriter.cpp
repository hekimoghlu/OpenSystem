/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
******************************************************************************
* Copyright (C) 1998-2012, International Business Machines Corporation and
* others. All Rights Reserved.
******************************************************************************
*
* File schriter.cpp
*
* Modification History:
*
*   Date        Name        Description
*  05/05/99     stephen     Cleaned up.
******************************************************************************
*/

#include "utypeinfo.h"  // for 'typeid' to work

#include "unicode/chariter.h"
#include "unicode/schriter.h"

U_NAMESPACE_BEGIN

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(StringCharacterIterator)

StringCharacterIterator::StringCharacterIterator()
  : UCharCharacterIterator(),
    text()
{
  // NEVER DEFAULT CONSTRUCT!
}

StringCharacterIterator::StringCharacterIterator(const UnicodeString& textStr)
  : UCharCharacterIterator(textStr.getBuffer(), textStr.length()),
    text(textStr)
{
    // we had set the input parameter's array, now we need to set our copy's array
    UCharCharacterIterator::text = this->text.getBuffer();
}

StringCharacterIterator::StringCharacterIterator(const UnicodeString& textStr,
                                                 int32_t textPos)
  : UCharCharacterIterator(textStr.getBuffer(), textStr.length(), textPos),
    text(textStr)
{
    // we had set the input parameter's array, now we need to set our copy's array
    UCharCharacterIterator::text = this->text.getBuffer();
}

StringCharacterIterator::StringCharacterIterator(const UnicodeString& textStr,
                                                 int32_t textBegin,
                                                 int32_t textEnd,
                                                 int32_t textPos)
  : UCharCharacterIterator(textStr.getBuffer(), textStr.length(), textBegin, textEnd, textPos),
    text(textStr)
{
    // we had set the input parameter's array, now we need to set our copy's array
    UCharCharacterIterator::text = this->text.getBuffer();
}

StringCharacterIterator::StringCharacterIterator(const StringCharacterIterator& that)
  : UCharCharacterIterator(that),
    text(that.text)
{
    // we had set the input parameter's array, now we need to set our copy's array
    UCharCharacterIterator::text = this->text.getBuffer();
}

StringCharacterIterator::~StringCharacterIterator() {
}

StringCharacterIterator&
StringCharacterIterator::operator=(const StringCharacterIterator& that) {
    UCharCharacterIterator::operator=(that);
    text = that.text;
    // we had set the input parameter's array, now we need to set our copy's array
    UCharCharacterIterator::text = this->text.getBuffer();
    return *this;
}

bool
StringCharacterIterator::operator==(const ForwardCharacterIterator& that) const {
    if (this == &that) {
        return true;
    }

    // do not call UCharCharacterIterator::operator==()
    // because that checks for array pointer equality
    // while we compare UnicodeString objects

    if (typeid(*this) != typeid(that)) {
        return false;
    }

    const StringCharacterIterator&    realThat = static_cast<const StringCharacterIterator&>(that);

    return text == realThat.text
        && pos == realThat.pos
        && begin == realThat.begin
        && end == realThat.end;
}

StringCharacterIterator*
StringCharacterIterator::clone() const {
    return new StringCharacterIterator(*this);
}

void
StringCharacterIterator::setText(const UnicodeString& newText) {
    text = newText;
    UCharCharacterIterator::setText(text.getBuffer(), text.length());
}

void
StringCharacterIterator::getText(UnicodeString& result) {
    result = text;
}
U_NAMESPACE_END
