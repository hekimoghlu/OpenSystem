/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
// Policies.h
//
#ifndef _SECURITY_POLICY_H_
#define _SECURITY_POLICY_H_

#include <Security/SecPolicy.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_client/tpclient.h>
#include <security_utilities/seccfobject.h>
#include "SecCFTypes.h"

namespace Security
{

namespace KeychainCore
{

using namespace CssmClient;

//
// A Policy[Impl] represents a particular
// CSSM "policy" managed by a particular TP.
//
class Policy : public SecCFObject
{
	NOCOPY(Policy)
public:
	SECCFFUNCTIONS(Policy, SecPolicyRef, errSecInvalidItemRef, gTypes().Policy)

    Policy(TP supportingTp, const CssmOid &policyOid);

public:
    virtual ~Policy() _NOEXCEPT;

    TP &tp()							{ return mTp; }
    const TP &tp() const				{ return mTp; }
    const CssmOid &oid() const			{ return mOid; }
    const CssmData &value() const		{ return mValue; }
	CssmOwnedData &value()				{ return mValue; }

    void setValue(const CssmData &value);
	void setProperties(CFDictionaryRef properties);
	CFDictionaryRef properties();

    bool operator < (const Policy& other) const;
    bool operator == (const Policy& other) const;

private:
    TP					mTp;			// TP module for this Policy
    CssmAutoData		mOid;			// OID for this policy
    CssmAutoData		mValue;			// value for this policy
    CssmAutoData		mAuxValue;		// variable-length value data for this policy
	Mutex				mMutex;
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_POLICY_H_
