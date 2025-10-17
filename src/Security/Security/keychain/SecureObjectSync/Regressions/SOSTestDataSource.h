/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#ifndef _SEC_SOSTestDataSource_H_
#define _SEC_SOSTestDataSource_H_

#include "keychain/SecureObjectSync/SOSDataSource.h"

extern CFStringRef sSOSDataSourceErrorDomain;

enum {
    kSOSDataSourceObjectMallocFailed = 1,
    kSOSDataSourceAddDuplicateEntry,
    kSOSDataSourceObjectNotFoundError,
    kSOSDataSourceAccountCreationFailed,
};

//
// MARK: Data Source Functions
//
SOSDataSourceRef SOSTestDataSourceCreate(void);

CFMutableDictionaryRef SOSTestDataSourceGetDatabase(SOSDataSourceRef data_source);

SOSMergeResult SOSTestDataSourceAddObject(SOSDataSourceRef data_source, SOSObjectRef object, CFErrorRef *error);
bool SOSTestDataSourceDeleteObject(SOSDataSourceRef data_source, CFDataRef key, CFErrorRef *error);

//
// MARK: Data Source Factory Functions
//

SOSDataSourceFactoryRef SOSTestDataSourceFactoryCreate(void);
void SOSTestDataSourceFactorySetDataSource(SOSDataSourceFactoryRef factory, CFStringRef name, SOSDataSourceRef ds);

SOSObjectRef SOSDataSourceCreateGenericItemWithData(SOSDataSourceRef ds, CFStringRef account, CFStringRef service, bool is_tomb, CFDataRef data);
SOSObjectRef SOSDataSourceCreateGenericItem(SOSDataSourceRef ds, CFStringRef account, CFStringRef service);
SOSObjectRef SOSDataSourceCreateV0EngineStateWithData(SOSDataSourceRef ds, CFDataRef engineStateData);

SOSObjectRef SOSDataSourceCopyObject(SOSDataSourceRef ds, SOSObjectRef match, CFErrorRef *error);

#endif /* _SEC_SOSTestDataSource_H_ */
