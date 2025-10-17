/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
// PrimaryKey.cpp
//

#include "PrimaryKey.h"

using namespace KeychainCore;
using namespace CssmClient;


PrimaryKeyImpl::PrimaryKeyImpl(const CSSM_DATA &data)
: CssmDataContainer(data.Data, data.Length), mMutex(Mutex::recursive)
{

//@@@ do bounds checking here, throw if invalid

}

PrimaryKeyImpl::PrimaryKeyImpl(const DbAttributes &primaryKeyAttrs) : mMutex(Mutex::recursive)
{
	Length = sizeof(uint32);
	for (uint32 ix = 0; ix < primaryKeyAttrs.size(); ++ix)
	{
		if (primaryKeyAttrs.at(ix).size() == 0)
			MacOSError::throwMe(errSecInvalidKeychain);

		Length += sizeof(uint32) + primaryKeyAttrs.at(ix).Value[0].Length;
	}

	// Careful with exceptions
	Data = mAllocator.alloc<uint8>((UInt32)Length);
	uint8 *p = Data;

	putUInt32(p, primaryKeyAttrs.recordType());
	for (uint32 ix = 0; ix < primaryKeyAttrs.size(); ++ix)
	{
		UInt32 len = (UInt32)primaryKeyAttrs.at(ix).Value[0].Length;
		putUInt32(p, len);
		memcpy(p, primaryKeyAttrs.at(ix).Value[0].Data, len);
		p += len;
	}
}

CssmClient::DbCursor
PrimaryKeyImpl::createCursor(const Keychain &keychain) 
{
	StLock<Mutex>_(mMutex);
	DbCursor cursor(keychain->database());

	// @@@ Set up cursor to find item with this.
	uint8 *p = Data;
	uint32 left = (uint32)Length;
	if (left < sizeof(*p))
		MacOSError::throwMe(errSecNoSuchAttr); // XXX Not really but whatever.

	CSSM_DB_RECORDTYPE rt = getUInt32(p, left);
	const CssmAutoDbRecordAttributeInfo &infos = keychain->primaryKeyInfosFor(rt);

	cursor->recordType(rt);
	cursor->conjunctive(CSSM_DB_AND);
	for (uint32 ix = 0; ix < infos.size(); ++ix)
	{
		uint32 len = getUInt32(p, left);

		if (left < len)
			MacOSError::throwMe(errSecNoSuchAttr); // XXX Not really but whatever.

		CssmData value(p, len);
		left -= len;
		p += len;

		cursor->add(CSSM_DB_EQUAL, infos.at(ix), value);
	}

	return cursor;
}


void
PrimaryKeyImpl::putUInt32(uint8 *&p, uint32 value)
{
	*p++ = (value >> 24);
	*p++ = (value >> 16) & 0xff;
	*p++ = (value >> 8) & 0xff;
	*p++ = value & 0xff;
}

uint32
PrimaryKeyImpl::getUInt32(uint8 *&p, uint32 &left) const
{
	if (left < sizeof(uint32))
		MacOSError::throwMe(errSecNoSuchAttr); // XXX Not really but whatever.


	// @@@ Assumes data written in big endian.
	uint32 value = (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + p[3];
	p += sizeof(uint32);
	left -= sizeof(uint32);
	return value;
}



CSSM_DB_RECORDTYPE
PrimaryKeyImpl::recordType() const
{
	uint8 *data = Data;
	uint32 length = (uint32)Length;
	return getUInt32(data, length);
}
