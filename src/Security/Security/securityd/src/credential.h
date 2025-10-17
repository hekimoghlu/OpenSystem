/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#ifndef _H_CREDENTIAL
#define _H_CREDENTIAL

#include <security_utilities/refcount.h>
#include <CoreFoundation/CFDate.h>
#include <set>

namespace Authorization {
    
    // There should be an abstract base class for Credential so we can have 
    // different kinds, e.g., those associated with smart-card auth, or those
    // not requiring authentication as such at all.  (<rdar://problem/6556724>)

/* Credentials are less than comparable so they can be put in sets or maps. */
class CredentialImpl : public RefCount
{
public:
		CredentialImpl();
        CredentialImpl(const uid_t uid, const string &username, const string &realname, bool shared);
        CredentialImpl(const string &username, const string &password, bool shared);
		CredentialImpl(const string &right, bool shared);
        ~CredentialImpl();

        bool operator < (const CredentialImpl &other) const;

        // Returns true if this credential should be shared.
        bool isShared() const;

        // Merge with other
        void merge(const CredentialImpl &other);

        // The time at which this credential was obtained.
        CFAbsoluteTime creationTime() const;

        // Return true iff this credential is valid.
        bool isValid() const;

        // Make this credential invalid.
        void invalidate();

        // We could make Rule a friend but instead we just expose this for now
        inline uid_t uid() const { return mUid; }
        inline const string& name() const { return mName; }
        inline const string& realname() const { return mRealName; }
        inline bool isRight() const { return mRight; }
    
private:
        bool mShared;       // credential is shared
        bool mRight;            // is least-privilege credential


        // Fields below are not used by less-than operator

        // The user that provided his password.
        uid_t mUid;
        string mName;
        string mRealName;

        CFAbsoluteTime mCreationTime;
        bool mValid;
};

/* Credentials are less than comparable so they can be put in sets or maps. */
class Credential : public RefPointer<CredentialImpl>
{
public:
        Credential();
        Credential(CredentialImpl *impl);
        Credential(const uid_t uid, const string &username, const string &realname, bool shared);
        Credential(const string &username, const string &password, bool shared);
		Credential(const string &right, bool shared);		
        ~Credential();

        bool operator < (const Credential &other) const;
};

typedef set<Credential> CredentialSet;

} // namespace Authorization

#endif // _H_CREDENTIAL
