/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
// SelectionPredicate.cpp
//

#include "SelectionPredicate.h"

SelectionPredicate::SelectionPredicate(const MetaRecord &inMetaRecord,
									   const CSSM_SELECTION_PREDICATE &inPredicate)
:	mMetaAttribute(inMetaRecord.metaAttribute(inPredicate.Attribute.Info)),
	mDbOperator(inPredicate.DbOperator)
{
	// Make sure that the caller specified the attribute values in the correct format.
	if (inPredicate.Attribute.Info.AttributeFormat != mMetaAttribute.attributeFormat())
		CssmError::throwMe(CSSMERR_DL_INCOMPATIBLE_FIELD_FORMAT);

	// XXX See ISSUES
	if (inPredicate.Attribute.NumberOfValues != 1)
		CssmError::throwMe(CSSMERR_DL_UNSUPPORTED_QUERY);

	mData = inPredicate.Attribute.Value[0];
	mValue = mMetaAttribute.createValue(mData);
}

SelectionPredicate::~SelectionPredicate()
{
	delete mValue;
}

bool
SelectionPredicate::evaluate(const ReadSection &rs) const
{
    return mMetaAttribute.evaluate(mValue, rs, mDbOperator);
}
