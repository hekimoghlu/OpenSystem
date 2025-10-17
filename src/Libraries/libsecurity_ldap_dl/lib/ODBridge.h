/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#ifndef _OD_BRIDGE_H_
#define _OD_BRIDGE_H_  1

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <OpenDirectory/OpenDirectory.h>
#include <dispatch/dispatch.h>
#include <Security/cssmapi.h>
#include <Security/cssmapple.h>
#include <Security/cssmerr.h>

// Query results are stored in this.
typedef struct ODdl_results {
	CSSM_DB_RECORDTYPE	recordid;
	ODQueryRef			query;
	CFStringRef			searchString;
	CFIndex				currentRecord;
	CFMutableArrayRef	certificates;
	dispatch_semaphore_t results_done;
	dispatch_queue_t	result_modifier_queue;
} *ODdl_results_handle;


// Oh how the mighty have fallen - had to get out of Dodge with one of these ...  once.
class DirectoryServiceException
{
protected:
	long mResult;
	
public:
	DirectoryServiceException (CFErrorRef result) : mResult (CFErrorGetCode(result)) {}
	
	long GetResult () {return mResult;}
};


class DirectoryService
{
protected:
	char						*db_name;
	ODNodeRef					node;
	dispatch_queue_t			query_dispatch_queue;	// Queue to use for queries
	CFMutableArrayRef			all_open_queries;
	
public:
	DirectoryService ();
	~DirectoryService ();
	long long int getNextRecordID();
	ODdl_results_handle makeNewDSQuery();
	ODdl_results_handle translate_cssm_query_to_OD_query(const CSSM_QUERY *Query, CSSM_RETURN *error);
	CFDataRef getNextCertFromResults(ODdl_results_handle results);
};


#endif /* !_OD_BRIDGE_H_ */
