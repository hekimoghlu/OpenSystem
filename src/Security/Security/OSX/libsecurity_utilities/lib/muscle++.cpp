/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
// C++ gate to "Muscle" smartcard interface layer
//
#include "muscle++.h"
#include <security_utilities/debugging.h>

#if TARGET_OS_OSX

namespace Security {
namespace Muscle {


//
// PCSC domain errors
//
Error::Error(MSC_RV err) : error(err)
{
    SECURITY_EXCEPTION_THROW_OTHER(this, err, (char *)"muscle");
    secnotice("security_exception", "muscle: %d", err);
}


const char *Error::what() const _NOEXCEPT
{
	return msc_error(error);
}


void Error::throwMe(MSC_RV err)
{
	throw Error(err);
}


OSStatus Error::osStatus() const
{
	return -1;	//@@@ preliminary
}

int Error::unixError() const
{
	return EINVAL;  //@@@ preliminary
}


//
// Open a connection with PCSC layer information.
// The ReaderState fields required are the slot name and the ATR.
//
Connection::Connection()
	: mIsOpen(false), mCurrentTransaction(NULL)
{
}

Connection::~Connection()
{
	assert(!mCurrentTransaction);
	close();
}

void Connection::open(const PCSC::ReaderState &reader, unsigned share)
{
	// fill in the minimum needed to identify the card
	MSCTokenInfo info;
	
	// set slot name in info
	strncpy(info.slotName, reader.name(), MAX_READERNAME);
	
	// set ATR in info
    if (reader.length() > MAX_ATR_SIZE) {
        Error::throwMe(MSC_INVALID_PARAMETER);
    }
    
	memcpy(info.tokenId, reader.data(), reader.length());
	info.tokenIdLength = (MSCULong32)reader.length();
	
	// establish Muscle-level connection to card
	Error::check(::MSCEstablishConnection(&info, share, NULL, 0, this));
	mIsOpen = true;
	secinfo("muscle", "%p opened %s", this, info.slotName);
	
	// pull initial status
	updateStatus();
}

void Connection::close()
{
	if (mIsOpen) {
		secinfo("muscle", "%p closing", this);
		Error::check(::MSCReleaseConnection(this, SCARD_LEAVE_CARD));
		mIsOpen = false;
	}
}


void Connection::begin(Transaction *trans)
{
	assert(!mCurrentTransaction);
	Error::check(::MSCBeginTransaction(this));
	secinfo("muscle", "%p start transaction %p", this, trans);
	mCurrentTransaction = trans;
}

void Connection::end(Transaction *trans)
{
	assert(trans == mCurrentTransaction);
	secinfo("muscle", "%p end transaction %p", this, trans);
	Error::check(::MSCEndTransaction(this, SCARD_LEAVE_CARD));
	mCurrentTransaction = NULL;
}


//
// Update card status (cached in the Connection object)
//
void Connection::updateStatus()
{
	Error::check(::MSCGetStatus(this, this));
}


//
// Get all items off the card
//
template <class Info, class Item, MSC_RV (*list)(MSCTokenConnection *, MSCUChar8, Info *)>
static void get(Connection *conn, Connection::ItemSet &items)
{
	Info info;
	MSCUChar8 seq = MSC_SEQUENCE_RESET;
	for (;;) {
		switch (MSC_RV rc = list(conn, seq, &info)) {
		case MSC_SEQUENCE_END:
			return;
		case MSC_SUCCESS:
			items.insert(new Item(info));
			seq = MSC_SEQUENCE_NEXT;
			break;
		default:
			Error::throwMe(rc);
		}
	}
}

void Connection::getItems(ItemSet &result, bool getKeys, bool getOthers)
{
	ItemSet items;
	if (getKeys)
		get<MSCKeyInfo, Key, MSCListKeys>(this, items);
	if (getOthers)
		get<MSCObjectInfo, Object, MSCListObjects>(this, items);
	items.swap(result);
}


//
// Transaction monitors
//
Transaction::Transaction(Connection &con)
    : connection(con)
{
    connection.begin(this);
}

Transaction::~Transaction()
{
    connection.end(this);
}


//
// ACLs (Muscle style)
//
static void aclForm(string &s, MSCUShort16 acl, int offset, char c)
{
	for (int n = 0; n < 5; n++) {
		char p = '-';
		switch (acl) {
		case MSC_AUT_ALL:		p = c; break;
		case MSC_AUT_NONE:		break;
		default:				if (acl & (MSC_AUT_PIN_0 << n)) p = c; break;
		}
		s[3 * n + offset] = p;
	}
}

string ACL::form(char ue) const
{
	string r = "---------------";
	aclForm(r, mRead, 0, 'r');
	aclForm(r, mWrite, 1, 'w');
	aclForm(r, mErase, 2, ue);
	return r;
}


//
// Keys and objects
//
CardItem::~CardItem()
{ /* virtual */ }


Key::Key(const MSCKeyInfo &info)
	: MSCKeyInfo(info)
{
	snprintf(mKeyName, sizeof(mKeyName), "K%d", id());
}


const ACL &Key::acl() const		{ return reinterpret_cast<const ACL &>(keyACL); }
ACL &Key::acl()					{ return reinterpret_cast<ACL &>(keyACL); }

const char *Key::name() const	{ return mKeyName; }
unsigned Key::size() const		{ return keySize; }

void Key::debugDump()
{
	printf("Key %d type %d size %d mode=0x%x dir=0x%x ACL %s\n",
		keyNum, keyType, keySize, mode(), operations(), acl().form('u').c_str());
}

const char *Object::name() const { return objectID; }
unsigned Object::size() const	{ return objectSize; }

const ACL &Object::acl() const	{ return reinterpret_cast<const ACL &>(objectACL); }
ACL &Object::acl()				{ return reinterpret_cast<ACL &>(objectACL); }

void Object::debugDump()
{
	printf("Object %s size %d ACL %s\n",
		objectID, objectSize, acl().form('e').c_str());
}


}	// namespace Muscle
}	// namespace Security

#endif //TARGET_OS_OSX
