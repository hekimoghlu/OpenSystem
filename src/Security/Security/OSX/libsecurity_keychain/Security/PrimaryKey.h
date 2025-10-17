/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
// PrimaryKey.h
//
#ifndef _SECURITY_PRIMARYKEY_H_
#define _SECURITY_PRIMARYKEY_H_

#include <security_cdsa_client/dlclient.h>
#include <security_keychain/Keychains.h>

namespace Security
{

namespace KeychainCore
{

class PrimaryKeyImpl : public CssmDataContainer
{
public:
    PrimaryKeyImpl(const CSSM_DATA &data);
    PrimaryKeyImpl(const CssmClient::DbAttributes &primaryKeyAttrs);
    ~PrimaryKeyImpl() {}

	void putUInt32(uint8 *&p, uint32 value);
	uint32 getUInt32(uint8 *&p, uint32 &left) const;

	CssmClient::DbCursor createCursor(const Keychain &keychain);

	CSSM_DB_RECORDTYPE recordType() const;
private:

protected:
	Mutex mMutex;
};


class PrimaryKey : public RefPointer<PrimaryKeyImpl>
{
public:
    PrimaryKey() {}
    PrimaryKey(PrimaryKeyImpl *impl) : RefPointer<PrimaryKeyImpl>(impl) {}
    PrimaryKey(const CSSM_DATA &data)
	: RefPointer<PrimaryKeyImpl>(new PrimaryKeyImpl(data)) {}
    PrimaryKey(const CssmClient::DbAttributes &primaryKeyAttrs)
	: RefPointer<PrimaryKeyImpl>(new PrimaryKeyImpl(primaryKeyAttrs)) {}
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_PRIMARYKEY_H_
