/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
// cssmmds - MDS interface for CSSM & friends  
//
#include "cssmmds.h"
#include <ctype.h>
#include <security_cdsa_client/dlquery.h>


//
// Construct a MdsComponent.
// This will perform an MDS lookup in the Common table
//
MdsComponent::MdsComponent(const Guid &guid) : mMyGuid(guid)
{
	using namespace MDSClient;
	Table<Common> common(mds());	// MDS "CDSA Common" table
	mCommon = common.fetch(Attribute("ModuleID") == mMyGuid, CSSMERR_CSSM_MDS_ERROR);
}

MdsComponent::~MdsComponent()
{
}
