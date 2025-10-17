/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
// sigblob - signature (Super)Blob types
//
#include "sigblob.h"
#include "CSCommon.h"


namespace Security {
namespace CodeSigning {


CFDataRef EmbeddedSignatureBlob::component(CodeDirectory::SpecialSlot slot) const
{
	const BlobCore *blob = this->find(slot);
	
	if (blob) {
		return blobData(slot, blob);
	}
	return NULL;
}
	
CFDataRef EmbeddedSignatureBlob::blobData(CodeDirectory::SpecialSlot slot, BlobCore const *blob)
{
	if (CodeDirectory::slotAttributes(slot) & cdComponentIsBlob) {
		return makeCFData(*blob);	// is a native Blob
	} else if (const BlobWrapper *wrap = BlobWrapper::specific(blob)) {
		return makeCFData(*wrap);
	} else {
		MacOSError::throwMe(errSecCSSignatureInvalid);
	}
}


void EmbeddedSignatureBlob::Maker::component(CodeDirectory::SpecialSlot slot, CFDataRef data)
{
	if (CodeDirectory::slotAttributes(slot) & cdComponentIsBlob)
		add(slot, reinterpret_cast<const BlobCore *>(CFDataGetBytePtr(data))->clone());
	else
		add(slot, BlobWrapper::alloc(CFDataGetBytePtr(data), CFDataGetLength(data)));
}


CFDictionaryRef EntitlementBlob::entitlements() const
{
	return makeCFDictionaryFrom(this->at<const UInt8 *>(sizeof(EntitlementBlob)),
		this->length() - sizeof(EntitlementBlob));
}

EntitlementDERBlob *EntitlementDERBlob::alloc(size_t length) {
	size_t blobLength = length + sizeof(BlobCore);
	if (blobLength < length) {
		// overflow
		return NULL;
	}

	EntitlementDERBlob *b = (EntitlementDERBlob *)malloc(blobLength);

	if (b != NULL) {
		b->BlobCore::initialize(kSecCodeMagicEntitlementDER, blobLength);
	}

	return b;
}

} // end namespace CodeSigning
} // end namespace Security
