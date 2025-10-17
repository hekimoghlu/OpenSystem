/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
/********************************************************************
 * COPYRIGHT: 
 * Copyright (c) 1997-2003, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/*   file name:  sfwdchit.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*/

#include "sfwdchit.h"
#include "unicode/ustring.h"
#include "unicode/unistr.h"
#include "uhash.h"
#include "cmemory.h"

// A hash code of kInvalidHashCode indicates that the has code needs
// to be computed. A hash code of kEmptyHashCode is used for empty keys
// and for any key whose computed hash code is kInvalidHashCode.
const int32_t SimpleFwdCharIterator::kInvalidHashCode = 0;
const int32_t SimpleFwdCharIterator::kEmptyHashCode = 1;

#if 0 // not used
SimpleFwdCharIterator::SimpleFwdCharIterator(const UnicodeString& s) {

    fHashCode = kInvalidHashCode;
    fLen = s.length();
    fStart = new char16_t[fLen];
    if(fStart == nullptr) {
        fBogus = true;
    } else {
        fEnd = fStart+fLen;
        fCurrent = fStart;
        fBogus = false;
        s.extract(0, fLen, fStart);          
    }
    
}
#endif

SimpleFwdCharIterator::SimpleFwdCharIterator(char16_t *s, int32_t len, UBool adopt) {

    fHashCode = kInvalidHashCode;

    fLen = len==-1 ? u_strlen(s) : len;

    if(adopt == false) {
        fStart = new char16_t[fLen];
        if(fStart == nullptr) {
            fBogus = true;
        } else {
            uprv_memcpy(fStart, s, fLen);
            fEnd = fStart+fLen;
            fCurrent = fStart;
            fBogus = false;
        }
    } else { // adopt = true
        fCurrent = fStart = s;
        fEnd = fStart + fLen;
        fBogus = false;
    }

}

SimpleFwdCharIterator::~SimpleFwdCharIterator() {
    delete[] fStart;
}

#if 0 // not used
bool SimpleFwdCharIterator::operator==(const ForwardCharacterIterator& that) const {
    if(this == &that) {
        return true;
    }
/*
    if(that->fHashCode != kInvalidHashCode && this->fHashCode = that->fHashCode) {
        return true;
    }

    if(this->fStart == that->fStart) {
        return true;
    }

    if(this->fLen == that->fLen && uprv_memcmp(this->fStart, that->fStart, this->fLen) {
        return true;
    }
*/
    return false;
}
#endif

int32_t SimpleFwdCharIterator::hashCode() const {
    if (fHashCode == kInvalidHashCode)
    {
        UHashTok key;
        key.pointer = fStart;
        const_cast<SimpleFwdCharIterator*>(this)->fHashCode = uhash_hashUChars(key);
    }
    return fHashCode;
}
        
UClassID SimpleFwdCharIterator::getDynamicClassID() const {
    return nullptr;
}

char16_t SimpleFwdCharIterator::nextPostInc() {
    if(fCurrent == fEnd) {
        return ForwardCharacterIterator::DONE;
    } else {
        return *(fCurrent)++;
    }
}
        
UChar32 SimpleFwdCharIterator::next32PostInc() {
    return ForwardCharacterIterator::DONE;
}
        
UBool SimpleFwdCharIterator::hasNext() {
    return fCurrent < fEnd;
}
