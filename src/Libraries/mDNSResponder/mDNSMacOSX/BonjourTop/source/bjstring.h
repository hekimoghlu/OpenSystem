/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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

//
//  bjstring.h
//  TestTB
//
//  Created by Terrin Eager on 9/26/12.
//
//

#ifndef __TestTB__bjstring__
#define __TestTB__bjstring__

#include <iostream>
#include "bjtypes.h"

class BJString
{

public:
    BJString();
    BJString(const BJString& scr);
    BJString(const char* str);
    virtual ~BJString();

    BJString& operator=(const char* str);
    BJString& operator=(const BJString& str);
    bool operator==(const char* str);
    bool operator!=(const char* str){return !operator==(str);}
    bool operator==(const BJString& str);
    bool operator!=(const BJString& str) {return !operator==(str);}
    bool operator<(const BJString& str) const;

    BJ_COMPARE Compare(const BJString& str);


    BJString& operator+=(const char* str);
    BJString& operator+=(const BJString& str);

    const char* GetBuffer() const;

    void Set(const char* str);
    void Set(const char* str,BJ_UINT32 len);

    void Append(const char* str, BJ_UINT32 len);

    bool Contains(const char* str);

    BJ_UINT32 GetUINT32();

    enum BJ_FORMAT_STYLE {BJSS_BYTE,BJSS_TIME} ;
    void Format(BJ_UINT64 number,BJ_FORMAT_STYLE style);

    BJ_UINT32 GetLength();

    BJ_UINT32 GetBufferLength(){return length;};

private:

    void Create(BJ_UINT32 len);
    char* buffer;
    BJ_UINT32 length;
};

#endif /* defined(__TestTB__bjstring__) */
