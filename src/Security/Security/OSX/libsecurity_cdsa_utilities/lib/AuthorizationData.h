/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
/*
 *  AuthorizationData.h
 *  Authorization
 */

#ifndef _H_AUTHORIZATIONDATA
#define _H_AUTHORIZATIONDATA  1

#include <Security/Authorization.h>
#include <Security/AuthorizationPlugin.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <CoreFoundation/CFDate.h>

#include <security_utilities/refcount.h>
#include <security_utilities/alloc.h>

#include <map>
#include <set>
#include <string>

// ptrdiff_t needed, so including STL type closest
#include <vector>

// @@@ Should consider making the various types better citizens by taking an Allocator, for now values are wiped.

namespace Authorization
{

class AuthValueOverlay : public AuthorizationValue
{
public:
	AuthValueOverlay(const string& stringValue) { length = stringValue.length(); data = const_cast<char *>(stringValue.c_str()); }
	AuthValueOverlay(UInt32 inLength, void *inData) { length = inLength; data = inData; }
};

class AuthValueRef;

class AuthValue : public RefCount
{
	friend class AuthValueRef;
private:
	AuthValue(const AuthValue& value) {}
protected:
	AuthValue(const AuthorizationValue &value);
	AuthValue(UInt32 length, void *data);
public:
    AuthValue &operator = (const AuthValue &other);
    ~AuthValue();
    void fillInAuthorizationValue(AuthorizationValue &value);
    const AuthorizationValue& value() const { return mValue; }
private:
    AuthorizationValue mValue;
    mutable bool mOwnsValue;
};

// AuthValueRef impl
class AuthValueRef : public RefPointer<AuthValue>
{
public:
    AuthValueRef(const AuthValue &value);
    AuthValueRef(const AuthorizationValue &value);
    AuthValueRef(UInt32 length, void *data);
};


// vector should become a member with accessors
class AuthValueVector : public vector<AuthValueRef>
{
public:
    AuthValueVector() {}
    ~AuthValueVector() {}

    AuthValueVector &operator = (const AuthorizationValueVector& valueVector);
};



class AuthItemRef;

class AuthItem : public RefCount
{
    friend class AuthItemRef;
private:
    AuthItem(const AuthItem& item);
protected:
    AuthItem(const AuthorizationItem &item);
    AuthItem(AuthorizationString name);
    AuthItem(AuthorizationString name, AuthorizationValue value);
    AuthItem(AuthorizationString name, AuthorizationValue value, AuthorizationFlags flags);

    bool operator < (const AuthItem &other) const;

public:
    AuthItem &operator = (const AuthItem &other);
    ~AuthItem();
    
    AuthorizationString name() const { return mName; }
    const AuthorizationValue& value() const { return mValue; }
	string stringValue() const { return string(static_cast<char *>(mValue.data), mValue.length); }
    AuthorizationFlags flags() const { return mFlags; }
	void setFlags(AuthorizationFlags inFlags) { mFlags = inFlags; };

private:
    AuthorizationString mName;
    AuthorizationValue mValue;
    AuthorizationFlags mFlags;
    mutable bool mOwnsName;
    mutable bool mOwnsValue;
	
public:
	bool getString(string &value);
	bool getCssmData(CssmAutoData &value);
};

class AuthItemRef : public RefPointer<AuthItem>
{
public:
    AuthItemRef(const AuthorizationItem &item);
    AuthItemRef(AuthorizationString name);
    AuthItemRef(AuthorizationString name, AuthorizationValue value, AuthorizationFlags flags = 0);

    bool operator < (const AuthItemRef &other) const
    {
        return **this < *other;
    }
};

// set should become a member with accessors
class AuthItemSet : public set<AuthItemRef>
{
public:
    AuthItemSet();
    ~AuthItemSet();
    AuthItemSet(const AuthorizationItemSet *item);
    AuthItemSet(const AuthItemSet& itemSet);

    AuthItemSet &operator = (const AuthorizationItemSet& itemSet);
    AuthItemSet &operator = (const AuthItemSet& itemSet);

public:
	AuthItem *find(const char *name);
};

class FindAuthItemByRightName
{
public:
    FindAuthItemByRightName(const char *find_name) : name(find_name) { }

    bool operator()( const AuthItemRef& authitem )
    {
        return (!strcmp(name, authitem->name()));
    }
    bool operator()( const AuthorizationItem* authitem )
    {
        return (!strcmp(name, authitem->name));
    }

private:
    const char *name;
};

}; // namespace Authorization

#endif /* ! _H_AUTHORIZATIONDATA */
