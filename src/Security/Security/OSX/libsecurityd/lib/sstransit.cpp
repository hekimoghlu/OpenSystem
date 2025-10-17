/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
// sstransit - Securityd client side transition support.
//
#include "sstransit.h"
#include <security_cdsa_client/cspclient.h>
#include <security_utilities/mach++.h>

namespace Security {
namespace SecurityServer {

using MachPlusPlus::check;
using MachPlusPlus::VMGuard;

//
// DataOutput helper.
// This happens "at the end" of a glue method, via the DataOutput destructor.
//
DataOutput::~DataOutput()
{
	// @@@ Why are we setting up a VMGuard if mData is NULL?
	VMGuard _(mData, mLength);
	if (mData)				// was assigned to; IPC returned OK
		if (mTarget) {		// output CssmData exists
			if (mTarget->data()) {	// caller provided buffer
				if (mTarget->length() < mLength)
					CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
				mTarget->length(mLength);	// no allocation; shorten buffer
			} else {	// allocate buffer
				*mTarget = CssmData(allocator.malloc(mLength), mLength);
			}
			memcpy(mTarget->data(), mData, mLength);
		}
}


//
// Copy an AccessCredentials for shipment.
// In addition, scan the samples for "special" database locking samples
// and translate certain items for safe shipment. Note that this overwrites
// part of the CssmList value (CSPHandle -> SS/KeyHandle), but we do it on
// the COPY, so that's okay.
//
DatabaseAccessCredentials::DatabaseAccessCredentials(const AccessCredentials *creds, Allocator &alloc)
	: Copier<AccessCredentials>(creds, alloc)
{
	if (creds) {
		for (uint32 n = 0; n < value()->samples().length(); n++) {
			TypedList sample = value()->samples()[n];
			sample.checkProper();
			switch (sample.type()) {
			case CSSM_SAMPLE_TYPE_KEYCHAIN_LOCK:
				sample.snip();			// skip sample type (snip() advances to next)
				sample.checkProper();
				if (sample.type() == CSSM_SAMPLE_TYPE_SYMMETRIC_KEY || 
					sample.type() == CSSM_SAMPLE_TYPE_ASYMMETRIC_KEY) {
					secinfo("SSclient", "key sample encountered");
					// proper form is sample[1] = DATA:CSPHandle, sample[2] = DATA:CSSM_KEY,
					// sample[3] = auxiliary data (not changed)
					if (sample.length() != 4
						|| sample[1].type() != CSSM_LIST_ELEMENT_DATUM
						|| sample[2].type() != CSSM_LIST_ELEMENT_DATUM
						|| sample[3].type() != CSSM_LIST_ELEMENT_DATUM)
						CssmError::throwMe(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
					mapKeySample(
								 sample[1].data(),
								 *sample[2].data().interpretedAs<CssmKey>(CSSM_ERRCODE_INVALID_SAMPLE_VALUE));
				}
				break;
			case CSSM_SAMPLE_TYPE_KEYCHAIN_CHANGE_LOCK:
				sample.snip();	// skip sample type
				sample.checkProper();
				if (sample.type() == CSSM_SAMPLE_TYPE_SYMMETRIC_KEY || 
					sample.type() == CSSM_SAMPLE_TYPE_ASYMMETRIC_KEY) {
					secinfo("SSclient", "key sample encountered");
					// proper form is sample[1] = DATA:CSPHandle, sample[2] = DATA:CSSM_KEY
					if (sample.length() != 3
						|| sample[1].type() != CSSM_LIST_ELEMENT_DATUM
						|| sample[2].type() != CSSM_LIST_ELEMENT_DATUM)
							CssmError::throwMe(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
					mapKeySample(
						sample[1].data(),
						*sample[2].data().interpretedAs<CssmKey>(CSSM_ERRCODE_INVALID_SAMPLE_VALUE));
				}
				break;
			default:
				break;
			}
		}
	}
}

void DatabaseAccessCredentials::mapKeySample(CssmData &cspHandleData, CssmKey &key)
{
	// We use a CSP passthrough to get the securityd key handle for a (reference) key.
	// We try the passthrough on everyone, since there's multiple CSP/DL modules
	// that play games with securityd, and we want to give everyone a chance to play.

	// @@@ can't use CssmClient (it makes its own attachments)
	CSSM_CC_HANDLE ctx;
    CSSM_CSP_HANDLE &cspHandle = *cspHandleData.interpretedAs<CSSM_CSP_HANDLE>(CSSM_ERRCODE_INVALID_SAMPLE_VALUE);
	CssmError::check(CSSM_CSP_CreatePassThroughContext(cspHandle, &key, &ctx));
	KeyHandle ssKey;
	CSSM_RETURN passthroughError =
		CSSM_CSP_PassThrough(ctx, CSSM_APPLESCPDL_CSP_GET_KEYHANDLE, NULL, (void **)&ssKey);
	CSSM_DeleteContext(ctx);	// ignore error
	switch (passthroughError) {
	case CSSM_OK:				// got the passthrough; rewrite the sample
		assert(sizeof(CSSM_CSP_HANDLE) >= sizeof(KeyHandle));	// future insurance
		cspHandle = ssKey;
        cspHandleData.length(sizeof(KeyHandle));
		secinfo("SSclient", "key sample mapped to key 0x%x", ssKey);
		return;
	case CSSMERR_CSP_INVALID_PASSTHROUGH_ID:
		return;		// CSP didn't understand the callback; leave the sample alone
	default:
		CssmError::throwMe(passthroughError);	// error
	}
}


//
// Inbound/outbound transit for the elaborate data-access attribute vectors
//
DataRetrieval::DataRetrieval(CssmDbRecordAttributeData *&attributes, Allocator &alloc)
	: Copier<CssmDbRecordAttributeData>(attributes),
	  mAllocator(alloc), mAttributes(attributes), mAddr(NULL), mBase(NULL), mLength(0)
{
}

DataRetrieval::~DataRetrieval()
{
	if (mAddr) {
		relocate(mAddr, mBase);
		if (mAttributes->size() != mAddr->size()) {
			secemergency("~DataRetrieval: size mismatch, %u != %u", mAttributes->size(), mAddr->size());
			abort();
		}
		
		// global (per-record) fields
		mAttributes->recordType(mAddr->recordType());
		mAttributes->semanticInformation(mAddr->semanticInformation());
		
		// transfer data values (but not infos, which we keep in the original vector)
		for (uint32 n = 0; n < mAttributes->size(); n++)
			mAttributes->at(n).copyValues(mAddr->at(n), mAllocator);
	}
}


} // namespace SecurityServer
} // end namespace Security
