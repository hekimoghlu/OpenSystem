/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
// cssmtrust - CSSM layer Trust (TP) related objects.
//
#include <security_cdsa_utilities/cssmtrust.h>
#include <security_utilities/debugging.h>


namespace Security {


//
// Cleanly release the memory in a TPEvidenceInfo
//
void TPEvidenceInfo::destroy(Allocator &allocator)
{
	// status code array
	if (codes() > 0 && StatusCodes)
		allocator.free(StatusCodes);

	//@@@ need to free unique id if present
}


}	// end namespace Security
