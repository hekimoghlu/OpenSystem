/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include <security_cdsa_utilities/AuthorizationData.h>
#include <security_cdsa_utilities/AuthorizationWalkers.h>
#include <security_cdsa_utilities/walkers.h>
#include <Security/checkpw.h>
#include <grp.h>
#include <pwd.h>


// checkpw() that uses provided struct passwd
extern "C"
{
int checkpw_internal( const struct passwd *pw, const char* password );
}


namespace Authorization {


AuthValueRef::AuthValueRef(const AuthValue &value) : 
	RefPointer<AuthValue>(new AuthValue(value)) {}

AuthValueRef::AuthValueRef(const AuthorizationValue &value) : 
	RefPointer<AuthValue>(new AuthValue(value)) {}

AuthValue::AuthValue(const AuthorizationValue &value) :
    mOwnsValue(false)
{
    mValue.length = value.length;
    mValue.data = value.data;
}

AuthValueRef::AuthValueRef(UInt32 length, void *data) : 
	RefPointer<AuthValue>(new AuthValue(length, data)) {}

AuthValue::AuthValue(UInt32 length, void *data) :
    mOwnsValue(true)
{
    mValue.length = length;
    mValue.data = new uint8_t[length];
    if (length)
        memcpy(mValue.data, data, length);
}

AuthValue::~AuthValue()
{
    if (mOwnsValue)
	{
		memset(mValue.data, 0, mValue.length);
        delete[] reinterpret_cast<uint8_t*>(mValue.data);
	}
}

AuthValue &
AuthValue::operator = (const AuthValue &other)
{
    if (mOwnsValue)
	{
		memset(mValue.data, 0 , mValue.length);
        delete[] reinterpret_cast<uint8_t*>(mValue.data);
	}
	
    mValue = other.mValue;
    mOwnsValue = other.mOwnsValue;
    other.mOwnsValue = false;
    return *this;
}

void
AuthValue::fillInAuthorizationValue(AuthorizationValue &value)
{
    value.length = mValue.length;
    value.data = mValue.data;
}

AuthValueVector &
AuthValueVector::operator = (const AuthorizationValueVector& valueVector)
{
    clear();
    for (unsigned int i=0; i < valueVector.count; i++)
        push_back(AuthValueRef(valueVector.values[i]));
    return *this;
}

AuthItem::AuthItem(const AuthorizationItem &item) :
    mFlags(item.flags),
    mOwnsName(true),
    mOwnsValue(true)
{
	if (!item.name)
		MacOSError::throwMe(errAuthorizationInternal);
	size_t nameLen = strlen(item.name) + 1;
	mName = new char[nameLen];
	memcpy(const_cast<char *>(mName), item.name, nameLen);

	mValue.length = item.valueLength;
	mValue.data = new uint8_t[item.valueLength];
	if (mValue.length)
		memcpy(mValue.data, item.value, item.valueLength);
}


AuthItem::AuthItem(AuthorizationString name) :
    mName(name),
    mFlags(0),
    mOwnsName(false),
    mOwnsValue(false)
{
    mValue.length = 0;
    mValue.data = NULL;
}

AuthItem::AuthItem(AuthorizationString name, AuthorizationValue value, AuthorizationFlags flags) :
    mFlags(flags),
    mOwnsName(true),
    mOwnsValue(true)
{
	if (!name)
		MacOSError::throwMe(errAuthorizationInternal);
	size_t nameLen = strlen(name) + 1;
	mName = new char[nameLen];
	memcpy(const_cast<char *>(mName), name, nameLen);

	mValue.length = value.length;
	mValue.data = new uint8_t[value.length];
	if (mValue.length)
		memcpy(mValue.data, value.data, value.length);
}

AuthItem::~AuthItem()
{
    if (mOwnsName)
        delete[] mName;
    if (mOwnsValue)
	{
		memset(mValue.data, 0, mValue.length);
        delete[] reinterpret_cast<uint8_t*>(mValue.data);
	}
}

bool
AuthItem::operator < (const AuthItem &other) const
{
    return strcmp(mName, other.mName) < 0;
}

AuthItem &
AuthItem::operator = (const AuthItem &other)
{
    if (mOwnsName)
        delete[] mName;
    if (mOwnsValue)
	{
		memset(mValue.data, 0, mValue.length);
        delete[] reinterpret_cast<uint8_t*>(mValue.data);
	}

    mName = other.mName;
    mValue = other.mValue;
    mFlags = other.mFlags;
    mOwnsName = other.mOwnsName;
    other.mOwnsName = false;
    mOwnsValue = other.mOwnsValue;
    other.mOwnsValue = false;
    return *this;
}

bool
AuthItem::getString(string &value)
{
	// if terminating NUL is included, ignore it
	size_t len = mValue.length;
	if (len > 0 && (static_cast<char*>(mValue.data)[len - 1] == 0))
		--len;
	value = string(static_cast<char*>(mValue.data), len);
	return true;
}

bool
AuthItem::getCssmData(CssmAutoData &value)
{
	value = CssmData(static_cast<uint8_t*>(mValue.data), mValue.length);
	return true;
}	


AuthItemRef::AuthItemRef(const AuthorizationItem &item) : RefPointer<AuthItem>(new AuthItem(item)) {}

AuthItemRef::AuthItemRef(AuthorizationString name) : RefPointer<AuthItem>(new AuthItem(name)) {}

AuthItemRef::AuthItemRef(AuthorizationString name, AuthorizationValue value, AuthorizationFlags flags) : RefPointer<AuthItem>(new AuthItem(name, value, flags)) {}


//
// AuthItemSet
//
AuthItemSet::AuthItemSet()
{
}

AuthItemSet::~AuthItemSet()
{
}

AuthItemSet &
AuthItemSet::operator = (const AuthorizationItemSet& itemSet)
{
    clear();

    for (unsigned int i=0; i < itemSet.count; i++)
        insert(AuthItemRef(itemSet.items[i]));

    return *this;
}

AuthItemSet&
AuthItemSet::operator=(const AuthItemSet& itemSet)
{
	std::set<AuthItemRef>::operator=(itemSet);	
	
	return *this;
}

AuthItemSet::AuthItemSet(const AuthorizationItemSet *itemSet)
{
	if (NULL != itemSet && NULL != itemSet->items)
	{
		for (unsigned int i=0; i < itemSet->count; i++)
			insert(AuthItemRef(itemSet->items[i]));
	}
}

AuthItemSet::AuthItemSet(const AuthItemSet& itemSet)
: std::set<AuthItemRef>(itemSet)
{
}

AuthItem *
AuthItemSet::find(const char *name)
{
	AuthItemSet::const_iterator found = find_if(this->begin(), this->end(), FindAuthItemByRightName(name) );
	if (found != this->end())
		return *found;
	
	return NULL;
}

}	// end namespace Authorization
