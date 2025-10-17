/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
//  bjStringtoStringMap.h
//  TestTB
//
//  Created by Terrin Eager on 12/21/12.
//
//

#ifndef __TestTB__bjStringtoStringMap__
#define __TestTB__bjStringtoStringMap__

#include <iostream>
#include "bjstring.h"
#include "LLRBTree.h"

class StringMapNode : public CRBNode<BJString>
{
public:
    StringMapNode();
    StringMapNode(BJString* pKey);
    ~StringMapNode();
    inline virtual BJ_COMPARE Compare(BJString* pKey);
    inline virtual void CopyNode(CRBNode* pSource);
    inline virtual void Init(){};
    inline virtual void Clear() {};


    BJString value;

};

class BJStringtoStringMap : public CLLRBTree<BJString, StringMapNode>
{
public:


};




#endif /* defined(__TestTB__bjStringtoStringMap__) */
