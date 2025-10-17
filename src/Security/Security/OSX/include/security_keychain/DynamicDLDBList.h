/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
 *  DynamicDLDBList.h
 */
#ifndef _SECURITY_DYNAMICDLDBLIST_H_
#define _SECURITY_DYNAMICDLDBLIST_H_

#include <security_cdsa_client/DLDBList.h>
#include <security_cdsa_client/cssmclient.h>

namespace Security
{

namespace KeychainCore
{

class DynamicDLDBList
{
public:
    DynamicDLDBList();
    ~DynamicDLDBList();

	const vector<DLDbIdentifier> &searchList();

protected:
	Mutex mMutex;
	bool _add(const Guid &guid, uint32 subserviceID, CSSM_SERVICE_TYPE subserviceType);
	bool _add(const DLDbIdentifier &);
	bool _remove(const Guid &guid, uint32 subserviceID, CSSM_SERVICE_TYPE subserviceType);
	bool _remove(const DLDbIdentifier &);
	bool _load();
	DLDbIdentifier dlDbIdentifier(const Guid &guid, uint32 subserviceID,
		CSSM_SERVICE_TYPE subserviceType);
	void callback(const Guid &guid, uint32 subserviceID,
		CSSM_SERVICE_TYPE subserviceType, CSSM_MODULE_EVENT eventType);

private:
	static CSSM_RETURN appNotifyCallback(const CSSM_GUID *guid, void *context,
		uint32 subserviceId, CSSM_SERVICE_TYPE subserviceType, CSSM_MODULE_EVENT eventType);

	vector<CssmClient::Module> mModules;
	typedef vector<DLDbIdentifier> SearchList;
	SearchList mSearchList;
    bool mSearchListSet;
};

} // end namespace KeychainCore

} // end namespace Security

#endif /* !_SECURITY_DYNAMICDLDBLIST_H_ */
