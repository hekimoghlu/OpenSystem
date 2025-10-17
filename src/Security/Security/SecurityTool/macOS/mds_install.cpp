/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "security_tool.h"
#include "mds_install.h"
#include <security_cdsa_client/mdsclient.h>

int
mds_install(int argc, char * const *argv)
{
	if(argc != 1) {
		/* crufty "show usage" return code */
		return SHOW_USAGE_MESSAGE;
	}

	try {
		MDSClient::mds().install();
	}
	catch(const CssmError &err) {
		cssmPerror("MDS_Install", err.error);
		return -1;
	}
	catch(...) {
		/* should never happen */
		fprintf(stderr, "Unexpected error on MDS_Install\n");
		return -1;
	}
	return 0;
}
