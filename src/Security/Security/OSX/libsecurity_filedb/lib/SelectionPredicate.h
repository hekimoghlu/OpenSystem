/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
// SelectionPredicate.h
//

#ifndef _H_APPLEDL_SELECTIONPREDICATE
#define _H_APPLEDL_SELECTIONPREDICATE

#include "MetaRecord.h"
#include <memory>

namespace Security
{

class SelectionPredicate
{
    NOCOPY(SelectionPredicate)
	
public:
    SelectionPredicate(const MetaRecord &inMetaRecord,
    				   const CSSM_SELECTION_PREDICATE &inPredicate);
	~SelectionPredicate();
	
    bool evaluate(const ReadSection &inReadSection) const;
	
private:
    const MetaAttribute &mMetaAttribute;
    CSSM_DB_OPERATOR mDbOperator;
	CssmDataContainer mData;
	DbValue *mValue;
};

} // end namespace Security

#endif // _H_APPLEDL_SELECTIONPREDICATE
